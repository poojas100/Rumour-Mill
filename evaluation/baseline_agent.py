"""
evaluation/baseline_agent.py
Improved rule-based baseline — safer, smarter, non-deterministic.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.rumor_env import RumorMillEnv
from environment.reward import SOURCE_RELIABILITY, SCENARIO_CORRECT_DECISION


# ✅ Safer source selection
SAFE_SOURCES = [
    s for s, score in SOURCE_RELIABILITY.items()
    if score >= 0.6
]

BLOCKED_SOURCES = {"gossip", "spinner"}

MAX_QUERIES = 3


def _count_reliable_sources(confirmed_sources: list) -> int:
    return sum(1 for s in confirmed_sources if SOURCE_RELIABILITY.get(s, 0) > 0.7)


def _detect_event(obs, signal_log: list) -> str:
    """Detect scenario from message content + signal log."""
    all_text = " ".join(obs.messages + obs.reddit_posts).lower()

    # Office
    if any(w in all_text for w in ["layoff", "cut", "engineering", "fired"]):
        return "layoffs"
    if any(w in all_text for w in ["budget", "revenue", "freeze", "quarter"]):
        return "revenue_miss"
    if any(w in all_text for w in ["promotion", "candidate"]):
        return "promotion_politics"
    if any(w in all_text for w in ["acqui", "merger", "deal"]):
        return "acquisition"
    if any(w in all_text for w in ["launch", "churn", "pivot"]):
        return "product_launch_fail"
    if any(w in all_text for w in ["ceo", "cto", "leaving"]):
        return "leadership_change"

    # Politics
    if any(w in all_text for w in ["election", "fraud", "vote"]):
        return "election_fraud"

    # Crypto
    if any(w in all_text for w in ["rug", "token", "defi"]):
        return "rug_pull"

    # Fallback
    signals = [s["type"] for s in signal_log]
    if "negative" in signals:
        return "layoffs"

    return "unknown"


def baseline_policy(obs, confirmed_sources: list, signal_log: list, decided: bool) -> dict:
    """Improved decision logic."""

    if decided:
        return {"type": "wait"}

    day = obs.day
    signals = [s["type"] for s in signal_log]

    has_negative = "negative" in signals
    has_positive = "positive" in signals
    contradictory = has_negative and has_positive

    reliable_count = _count_reliable_sources(confirmed_sources)
    event_guess = _detect_event(obs, signal_log)

    # 🔹 Phase 1 — Gather (short and efficient)
    gather_until = 3

    if day <= gather_until:

        # Resolve contradictions actively
        if contradictory:
            for source in SAFE_SOURCES:
                if source not in confirmed_sources:
                    return {
                        "type": "message_character",
                        "target": source,
                        "content": "Can you confirm this?",
                    }

        # Query only safe sources
        if len(confirmed_sources) < MAX_QUERIES:
            for source in SAFE_SOURCES:
                if source not in confirmed_sources:
                    return {
                        "type": "message_character",
                        "target": source,
                        "content": "What have you heard recently?",
                    }

        return {"type": "wait"}

    # 🔹 Phase 2 — Early decision (smart)
    if reliable_count >= 2 and not contradictory:
        correct = SCENARIO_CORRECT_DECISION.get(event_guess)
        if correct:
            return {"type": "make_decision", "decision": correct}

    # 🔹 Hard cap on querying
    if len(confirmed_sources) >= MAX_QUERIES:
        correct = SCENARIO_CORRECT_DECISION.get(event_guess)
        if correct:
            return {"type": "make_decision", "decision": correct}

    # 🔹 Signal fallback
    if has_negative and reliable_count >= 1:
        return {"type": "make_decision", "decision": "warn_team_quietly"}

    if not has_negative and not has_positive:
        return {"type": "make_decision", "decision": "ignore"}

    return {"type": "make_decision", "decision": "wait_for_more_signals"}


def run_single_episode(verbose=True) -> float:
    env = RumorMillEnv()
    obs = env.reset()

    total_reward = 0.0
    done = False
    decided = False

    if verbose:
        gt = env.ground_truth
        domain = gt.get("domain", "office")
        event = gt.get("event_type") or gt.get("event", "?")
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
            if action.get("target"):
                action_str += f" → {action['target']}"
            if action.get("decision"):
                action_str += f" → {action['decision']}"

            print(
                f"  Day {obs.day-1} | {action_str:50s} | "
                f"reward={obs.reward:+.4f} | social={obs.social_capital:.0f}"
            )

    if verbose and obs.ground_truth_revealed:
        gt = obs.ground_truth_revealed
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

    avg = sum(scores) / len(scores)
    best = max(scores)
    worst = min(scores)
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
        "avg": avg,
        "best": best,
        "worst": worst,
        "scores": scores,
        "above_zero_pct": 100 * above_zero / n,
    }


if __name__ == "__main__":
    run_single_episode(verbose=True)
    run_multiple_episodes(n=20)