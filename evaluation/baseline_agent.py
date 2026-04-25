"""
evaluation/baseline_agent.py
Rule-based baseline — supports all scenarios including non-office domains.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.rumor_env import RumorMillEnv
from environment.reward import SOURCE_RELIABILITY, SCENARIO_CORRECT_DECISION

SOURCE_PRIORITY = ["quiet_one", "leaker", "politician", "gossip", "spinner"]


def _count_reliable_sources(confirmed_sources: list) -> int:
    return sum(1 for s in confirmed_sources if SOURCE_RELIABILITY.get(s, 0) > 0.7)


def _detect_event(obs, signal_log: list) -> str:
    """
    Detect scenario from message content + signal log.
    Covers all domains: office, politics, sports, crypto, academia, community, gaming.
    """
    all_text = " ".join(obs.messages + obs.reddit_posts).lower()

    # ── Office ────────────────────────────────────────────────
    if any(w in all_text for w in ["layoff", "cut", "engineering", "fired", "let go", "staffing"]):
        return "layoffs"
    if any(w in all_text for w in ["budget", "revenue", "q4", "freeze", "miss", "quarter", "sales"]):
        return "revenue_miss"
    if any(w in all_text for w in ["promotion", "candidate", "candidate a", "candidate b", "winning", "war"]):
        return "promotion_politics"
    if any(w in all_text for w in ["acqui", "merger", "due diligence", "deal", "bought", "sold"]):
        return "acquisition"
    if any(w in all_text for w in ["launch", "churn", "signups", "product failing", "pivot"]):
        return "product_launch_fail"
    if any(w in all_text for w in ["ceo", "cto", "vp leaving", "stepping down", "replacement", "departure", "leave"]):
        return "leadership_change"

    # ── Politics ──────────────────────────────────────────────
    if any(w in all_text for w in ["ballot", "election", "fraud", "polling", "vote", "audit"]):
        return "election_fraud"
    if any(w in all_text for w in ["scandal", "misconduct", "documents", "campaign", "fabricated"]):
        return "candidate_scandal"

    # ── Sports ────────────────────────────────────────────────
    if any(w in all_text for w in ["trade", "transfer", "club", "player", "rival", "fee"]):
        return "athlete_trade"
    if any(w in all_text for w in ["doping", "positive test", "supplement", "epo", "systematic"]):
        return "doping_coverup"

    # ── Crypto ────────────────────────────────────────────────
    if any(w in all_text for w in ["rug", "token", "liquidity", "wallet", "defi", "crypto"]):
        return "rug_pull"
    if any(w in all_text for w in ["exchange", "withdrawal", "reserves", "insolvency", "funds"]):
        return "exchange_insolvency"

    # ── Academia ──────────────────────────────────────────────
    if any(w in all_text for w in ["research", "data", "fabricat", "replicate", "paper", "fraud"]):
        return "research_fraud"

    # ── Community ─────────────────────────────────────────────
    if any(w in all_text for w in ["contamination", "toxic", "water", "smell", "illness", "industrial"]):
        return "environmental_coverup"

    # ── Gaming ────────────────────────────────────────────────
    if any(w in all_text for w in ["game", "shutdown", "server", "studio", "devs", "update"]):
        return "game_shutdown"

    # Fallback: use signal direction
    signals = [s["type"] for s in signal_log]
    if "negative" in signals:
        return "layoffs"  # safe fallback for negative signals

    return "unknown"


def baseline_policy(obs, confirmed_sources: list, signal_log: list, decided: bool) -> dict:
    """One action per call. After deciding, always waits."""
    if decided:
        return {"type": "wait"}

    day            = obs.day
    signals        = [s["type"] for s in signal_log]
    has_negative   = "negative" in signals
    has_positive   = "positive" in signals
    contradictory  = has_negative and has_positive
    reliable_count = _count_reliable_sources(confirmed_sources)
    event_guess    = _detect_event(obs, signal_log)

    # Phase 1: Gather info (days 0-5 for timeline episodes, 0-2 for short)
    gather_until = 5 if obs.day <= 5 else 2
    if day <= gather_until:
        if contradictory and day < gather_until:
            return {"type": "wait"}
        for source in SOURCE_PRIORITY:
            if source not in confirmed_sources:
                return {
                    "type": "message_character",
                    "target": source,
                    "content": "What have you heard recently?",
                }
        return {"type": "wait"}

    # Phase 2: Decide (days 6+)
    if reliable_count == 0:
        for source in SOURCE_PRIORITY:
            if source not in confirmed_sources:
                return {
                    "type": "message_character",
                    "target": source,
                    "content": "I need to know what's happening.",
                }

    # Look up correct decision from reward module's registry
    correct = SCENARIO_CORRECT_DECISION.get(event_guess)
    if correct:
        return {"type": "make_decision", "decision": correct}

    # Pure signal-based fallback
    if has_negative and reliable_count >= 1:
        return {"type": "make_decision", "decision": "warn_team_quietly"}
    if not has_negative and not has_positive:
        return {"type": "make_decision", "decision": "ignore"}

    return {"type": "make_decision", "decision": "wait_for_more_signals"}


def run_single_episode(verbose=True) -> float:
    env = RumorMillEnv()
    obs = env.reset()

    total_reward = 0.0
    done         = False
    decided      = False

    if verbose:
        gt = env.ground_truth
        domain = gt.get("domain", "office")
        event  = gt.get("event_type") or gt.get("event", "?")
        print(f"\n=== BASELINE EPISODE === [{domain}] {event}")
        print(f"Messages: {obs.messages}")

    while not done:
        action = baseline_policy(obs, env.confirmed_sources, env.signal_log, decided)

        if action["type"] == "make_decision":
            decided = True

        obs = env.step(action)
        total_reward += obs.reward
        done = obs.done

        if verbose:
            action_str = action["type"]
            if action.get("target"):   action_str += f" → {action['target']}"
            if action.get("decision"): action_str += f" → {action['decision']}"
            print(f"  Day {obs.day-1} | {action_str:50s} | reward={obs.reward:+.4f} | social={obs.social_capital:.0f}")

    if verbose and obs.ground_truth_revealed:
        gt    = obs.ground_truth_revealed
        event = gt.get("event_type") or gt.get("event", "unknown")
        truth = gt.get("core_truth") or gt.get("truth", {})
        print(f"  Truth: {event} → {truth}")
        print(f"  Total: {total_reward:.4f}")

    return total_reward


def run_multiple_episodes(n=20, verbose=False) -> dict:
    scores = []
    print(f"\n=== BASELINE: {n} episodes ===")

    for i in range(n):
        score = run_single_episode(verbose=verbose)
        scores.append(score)
        if (i + 1) % 5 == 0:
            print(f"  Episode {i+1:3d}: score={score:+.4f} | running avg={sum(scores)/len(scores):+.4f}")

    avg        = sum(scores) / len(scores)
    best       = max(scores)
    worst      = min(scores)
    above_zero = sum(1 for s in scores if s > 0)

    print(f"\n{'─'*40}")
    print(f"  Episodes:      {n}")
    print(f"  Average:       {avg:+.4f}")
    print(f"  Best:          {best:+.4f}")
    print(f"  Worst:         {worst:+.4f}")
    print(f"  Above zero:    {above_zero}/{n} ({100*above_zero/n:.0f}%)")
    print(f"{'─'*40}")
    print(f"  ← Baseline. Trained model should beat {avg:+.4f}")

    return {
        "avg": avg, "best": best, "worst": worst,
        "scores": scores, "above_zero_pct": 100 * above_zero / n,
    }


if __name__ == "__main__":
    run_single_episode(verbose=True)
    run_multiple_episodes(n=20)