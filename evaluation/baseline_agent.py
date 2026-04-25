"""
evaluation/baseline_agent.py
Rule-based baseline agent for comparison against trained LLM.

Usage:
    python -m evaluation.baseline_agent
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.rumor_env import RumorMillEnv
from environment.reward import SOURCE_RELIABILITY

SOURCE_PRIORITY = ["quiet_one", "leaker", "politician", "gossip", "spinner"]


def _count_reliable_sources(confirmed_sources: list) -> int:
    return sum(1 for s in confirmed_sources if SOURCE_RELIABILITY.get(s, 0) > 0.7)


def _detect_event(obs) -> str:
    all_text = " ".join(obs.messages + obs.reddit_posts).lower()
    if any(w in all_text for w in ["layoff", "cut", "engineering", "fired"]):
        return "layoffs"
    if any(w in all_text for w in ["budget", "revenue", "q4", "freeze", "miss"]):
        return "revenue_miss"
    if any(w in all_text for w in ["promotion", "candidate", "politics", "leadership"]):
        return "promotion_politics"
    return "unknown"


def baseline_policy(obs, confirmed_sources: list, signal_log: list, decided: bool) -> dict:
    # ✅ Never act again after making a decision — prevents streak penalty
    if decided:
        return {"type": "wait"}

    day            = obs.day
    signals        = [s["type"] for s in signal_log]
    has_negative   = "negative" in signals
    has_positive   = "positive" in signals
    contradictory  = has_negative and has_positive
    reliable_count = _count_reliable_sources(confirmed_sources)
    event_guess    = _detect_event(obs)

    # Phase 1: Gather (days 0-2)
    if day <= 2:
        if contradictory and day < 2:
            return {"type": "wait"}
        for source in SOURCE_PRIORITY:
            if source not in confirmed_sources:
                return {"type": "message_character", "target": source,
                        "content": "What have you heard recently?"}
        return {"type": "wait"}

    # Phase 2: Decide (days 3+)
    if reliable_count == 0:
        for source in SOURCE_PRIORITY:
            if source not in confirmed_sources:
                return {"type": "message_character", "target": source,
                        "content": "I need to know what's happening."}

    if event_guess == "layoffs":
        return {"type": "make_decision", "decision": "warn_team_quietly"}
    if event_guess == "revenue_miss":
        return {"type": "make_decision", "decision": "request_budget_freeze"}
    if event_guess == "promotion_politics":
        return {"type": "make_decision", "decision": "escalate_to_leadership"}
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
        print("\n=== BASELINE EPISODE ===")
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
            print(f"  Day {obs.day-1} | {action_str:45s} | reward={obs.reward:+.2f} | social={obs.social_capital:.0f}")

    if verbose and obs.ground_truth_revealed:
        event = obs.ground_truth_revealed.get("event", "unknown")  # ✅ fixed
        truth = obs.ground_truth_revealed.get("truth", {})
        print(f"  Total: {total_reward:.2f}")

    return total_reward


def run_multiple_episodes(n=20, verbose=False) -> dict:
    scores = []
    print(f"\n=== BASELINE: {n} episodes ===")

    for i in range(n):
        score = run_single_episode(verbose=verbose)
        scores.append(score)
        if (i + 1) % 5 == 0:
            print(f"  Episode {i+1:3d}: score={score:+.2f} | running avg={sum(scores)/len(scores):+.2f}")

    avg        = sum(scores) / len(scores)
    best       = max(scores)
    worst      = min(scores)
    above_zero = sum(1 for s in scores if s > 0)

    print(f"\n{'─'*40}")
    print(f"  Episodes:      {n}")
    print(f"  Average:       {avg:+.2f}")
    print(f"  Best:          {best:+.2f}")
    print(f"  Worst:         {worst:+.2f}")
    print(f"  Above zero:    {above_zero}/{n} ({100*above_zero/n:.0f}%)")
    print(f"{'─'*40}")
    print(f"  ← Baseline score. Trained model should beat {avg:+.2f}")

    return {"avg": avg, "best": best, "worst": worst,
            "scores": scores, "above_zero_pct": 100 * above_zero / n}


if __name__ == "__main__":
    run_single_episode(verbose=True)
    run_multiple_episodes(n=20)