"""
demo/sample_episodes.py

Short console output by default (one line per episode + summary).

Env vars:
  RUMOUR_VERBOSE=1     full step-by-step logs
  RUMOUR_STEPS=1       one-line action trace per episode (with default)
  RUMOUR_EPISODES=N    number of episodes (default 8)
  RUMOUR_QUIET=1       hide [inference_agent] import banner
  RUMOUR_AGENT_LOG=1   print scripted-agent decisions
"""

import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.rumor_env import RumorMillEnv
from environment.models import RumorAction
from demo.inference_agent import (
    build_veritarl_prompt,
    generate as agent_generate,
    get_agent_stats,
    new_episode as agent_new_episode,
)

CHARACTERS = ["spinner", "gossip", "quiet_one", "politician", "leaker"]
DECISIONS = [
    "warn_team_quietly", "wait_for_more_signals",
    "escalate_to_leadership", "request_budget_freeze", "ignore",
]

VERBOSE      = os.environ.get("RUMOUR_VERBOSE", "").strip() == "1"
SHOW_STEPS   = os.environ.get("RUMOUR_STEPS", "").strip() == "1"
NUM_EPISODES = int(os.environ.get("RUMOUR_EPISODES", "8"))


# ── action bridge ─────────────────────────────────────────────────

def _signals_to_decision(signals: list[str]) -> str:
    """Map the day's signals → one of the env's five decisions."""
    text = " ".join(signals).lower()
    if any(kw in text for kw in ["layoff", "cut", "fired", "engineering"]):
        return "warn_team_quietly"
    if any(kw in text for kw in ["revenue", "q4", "budget", "miss", "freeze"]):
        return "request_budget_freeze"
    if any(kw in text for kw in ["promotion", "politics", "leadership change", "escalat"]):
        return "escalate_to_leadership"
    if any(kw in text for kw in ["no issue", "fine", "overblown", "business operating"]):
        return "ignore"
    return "wait_for_more_signals"


def parse_action(raw: str, obs, consulted: set[str]) -> RumorAction:
    """
    Translate the trained model's ACTION / STEP / RATIONALE text (or the
    scripted agent's `wait` / `message X` / `decide Y`) into a RumorAction.
    """
    lower = raw.lower().strip()
    signals = list(obs.messages) + list(obs.reddit_posts)

    # --- trained model format: look for ACTION: ... ---
    m = re.search(r"action\s*:\s*(\w+)", lower)
    if m:
        act = m.group(1)
        if act in {"verify", "gather"}:
            for src in ["quiet_one", "leaker", "politician", "gossip", "spinner"]:
                if src not in consulted:
                    return RumorAction(
                        type="message_character",
                        target=src,
                        content=f"What have you heard? ({act})",
                    )
            return RumorAction(
                type="make_decision",
                decision=_signals_to_decision(signals),
            )
        if act == "wait":
            return RumorAction(type="wait")
        if act == "dismiss":
            return RumorAction(type="make_decision", decision="ignore")
        if act == "confirm":
            return RumorAction(
                type="make_decision",
                decision=_signals_to_decision(signals),
            )

    # --- scripted / legacy format: `message X` / `decide Y` / `wait` ---
    line = lower.split("\n", 1)[0]

    if "message" in line or "dm" in line or "ask" in line:
        target = next((c for c in CHARACTERS
                       if c in line or c.replace("_", " ") in line), None)
        if not target:
            target = next((c for c in CHARACTERS if c in lower), "quiet_one")
        return RumorAction(type="message_character", target=target, content=line)

    if any(w in line for w in ["decide", "warn", "escalate", "freeze", "ignore", "wait_for"]):
        decision = next((d for d in DECISIONS
                         if d in line or d.replace("_", " ") in line), None)
        if not decision:
            if   "warn"  in line: decision = "warn_team_quietly"
            elif "escal" in line: decision = "escalate_to_leadership"
            elif "freez" in line: decision = "request_budget_freeze"
            elif "ignor" in line: decision = "ignore"
            else:                 decision = "wait_for_more_signals"
        return RumorAction(type="make_decision", decision=decision)

    for c in CHARACTERS:
        if c in lower or c.replace("_", " ") in lower:
            return RumorAction(type="message_character", target=c, content=line)

    return RumorAction(type="wait")


# ── formatting helpers ────────────────────────────────────────────

def _action_label(a: RumorAction) -> str:
    s = a.type
    if a.target:   s += f" -> {a.target}"
    if a.decision: s += f" -> {a.decision}"
    return s


def _tiny(a: RumorAction) -> str:
    if a.type == "wait":
        return "wait"
    if a.type == "message_character" and a.target:
        return f"msg:{a.target}"
    if a.type == "make_decision" and a.decision:
        d = a.decision
        mapping = {
            "warn_team_quietly":      "dec:warn",
            "request_budget_freeze":  "dec:freeze",
            "escalate_to_leadership": "dec:escalate",
            "wait_for_more_signals":  "dec:waitinfo",
            "ignore":                 "dec:ignore",
        }
        return mapping.get(d, "dec:" + d[:12])
    return a.type[:10]


# ── episode loop ──────────────────────────────────────────────────

def run_episode(env, episode_num: int, total_eps: int) -> dict:
    agent_new_episode()
    obs = env.reset()
    stats = get_agent_stats()
    consulted: set[str] = set()
    total_reward = 0.0
    tiny_steps: list[str] = []
    first_action_text = ""

    if VERBOSE:
        print(f"\n{'='*48}")
        print(f"EPISODE {episode_num}/{total_eps}  epsilon={stats['epsilon']:.2f}")
        print(f"{'='*48}")
        for m in obs.messages:
            print(f"  {m}")
        for p in obs.reddit_posts:
            print(f"  reddit: {p}")

    for step in range(env.max_days):
        signals = list(obs.messages) + list(obs.reddit_posts)
        prompt  = build_veritarl_prompt(obs.day, obs.social_capital, signals)
        raw     = agent_generate(prompt, max_new_tokens=96)

        if step == 0:
            first_action_text = raw.split("\n", 1)[0]

        action = parse_action(raw, obs, consulted)
        if action.type == "message_character" and action.target:
            consulted.add(action.target)

        obs = env.step(action)
        total_reward += obs.reward
        tiny_steps.append(_tiny(action))

        if VERBOSE:
            first_line = raw.strip().split("\n", 1)[0] if raw.strip() else "(empty)"
            print(f"\n  step {step}  day {obs.day - 1}/{env.max_days}")
            print(f"    model: {first_line}")
            print(f"    ->    {_action_label(action)}")
            print(f"    r {obs.reward:+.3f}  sum {total_reward:+.3f}")
            if obs.dm_response:
                print(f"    DM: {obs.dm_response}")
            for m in obs.messages:
                print(f"    + {m}")

        if obs.done:
            gt = obs.ground_truth_revealed or {}
            event = gt.get("event_type") or gt.get("event", "?")
            rb = getattr(obs, "reward_breakdown", None) or {}
            correct = rb.get("correct_decision", False)
            ok = "yes" if correct else "no"

            if not VERBOSE:
                line = (f"  {episode_num:>2}/{total_eps}  {event:<22}  "
                        f"reward {total_reward:>+7.2f}  ok {ok}")
                if SHOW_STEPS:
                    line += "\n      " + " -> ".join(tiny_steps)
                print(line)
            else:
                truth = gt.get("core_truth") or gt.get("truth", {})
                print(f"\n  scenario: {event}")
                print(f"  truth: {truth}")
                print(f"  total {total_reward:+.3f}  correct {ok}")

            return {
                "episode":  episode_num,
                "event":    event,
                "reward":   total_reward,
                "correct":  correct,
                "first":    first_action_text,
            }

    if not VERBOSE:
        print(f"  {episode_num:>2}/{total_eps}  (no terminal state)  "
              f"reward {total_reward:>+7.2f}")
    return {
        "episode": episode_num, "event": "timeout",
        "reward":  total_reward, "correct": False,
        "first":   first_action_text,
    }


def print_summary(results: list[dict]) -> None:
    n = len(results)
    ok = sum(1 for r in results if r["correct"])
    total_r = sum(r["reward"] for r in results)
    stats = get_agent_stats()

    print()
    print(f"  {ok}/{n} episodes correct ({ok/n:.0%})   total reward {total_r:+.2f}")
    if stats["exploit_count"] + stats["explore_count"] > 0:
        print(f"  end epsilon {stats['epsilon']:.2f}   "
              f"explore share {stats['explore_rate']:.0%}")
    print()
    for r in results:
        mark = "y" if r["correct"] else " "
        print(f"    {r['episode']:>2}  {r['event']:<22}  "
              f"{r['reward']:>+7.2f}  [{mark}]")


if __name__ == "__main__":
    if not VERBOSE:
        print(
            "VeritaRL demo (compact). "
            "RUMOUR_VERBOSE=1 full logs | RUMOUR_STEPS=1 action trace | "
            "RUMOUR_AGENT_LOG=1 agent trace\n"
        )

    env = RumorMillEnv()
    results = []
    for ep in range(1, NUM_EPISODES + 1):
        results.append(run_episode(env, ep, NUM_EPISODES))

    print_summary(results)
