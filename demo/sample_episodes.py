"""
demo/sample_episodes.py

Default: very short console output (one line per episode + small summary).

  RUMOUR_VERBOSE=1     full step-by-step logs
  RUMOUR_STEPS=1       add one line of short action codes per episode (with default)
  RUMOUR_EPISODES=N    number of episodes (default 8)
  RUMOUR_QUIET=1       hide [inference_agent] import banner
  RUMOUR_AGENT_LOG=1   print scripted-agent [DEMO ...] lines each step
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.rumor_env import RumorMillEnv
from environment.models import RumorAction
from demo.inference_agent import generate as agent_generate
from demo.inference_agent import new_episode as agent_new_episode
from demo.inference_agent import get_agent_stats

CHARACTERS = ["spinner", "gossip", "quiet_one", "politician", "leaker"]
DECISIONS  = [
    "warn_team_quietly", "wait_for_more_signals",
    "escalate_to_leadership", "request_budget_freeze", "ignore",
]

VERBOSE = os.environ.get("RUMOUR_VERBOSE", "").strip() == "1"
SHOW_STEPS = os.environ.get("RUMOUR_STEPS", "").strip() == "1"
NUM_EPISODES = int(os.environ.get("RUMOUR_EPISODES", "8"))


def build_prompt(obs):
    msgs = "\n".join(f"  {m}" for m in obs.messages) or "  (no messages)"
    redd = "\n".join(f"  {p}" for p in obs.reddit_posts) or "  (none)"
    return f"""You are navigating rumours in a social network. Choose ONE action.

DAY {obs.day}/5  |  Reputation: {obs.social_capital:.0f}/100

MESSAGES:
{msgs}

REDDIT:
{redd}

Valid actions (copy one exactly, no explanation):
  wait
  message quiet_one
  message leaker
  message gossip
  message spinner
  message politician
  decide warn_team_quietly
  decide wait_for_more_signals
  decide escalate_to_leadership
  decide request_budget_freeze
  decide ignore

Examples:
  message quiet_one
  decide warn_team_quietly
  wait

Your single action (one line only):"""


def parse_action(raw):
    lines = [l.strip().lstrip("-*123456789. ") for l in raw.strip().split("\n")]
    lines = [l for l in lines if l]
    text  = lines[0].lower() if lines else ""

    if "message" in text or "dm" in text or "ask" in text:
        target = next((c for c in CHARACTERS if c in text or c.replace("_"," ") in text), None)
        if not target:
            target = next((c for c in CHARACTERS if c in raw.lower()), "quiet_one")
        return RumorAction(type="message_character", target=target, content=text)

    if any(w in text for w in ["decide", "warn", "escalate", "freeze", "ignore", "wait_for"]):
        decision = next((d for d in DECISIONS if d.replace("_"," ") in text or d in text), None)
        if not decision:
            if "warn"    in text: decision = "warn_team_quietly"
            elif "escal" in text: decision = "escalate_to_leadership"
            elif "freez" in text: decision = "request_budget_freeze"
            elif "ignor" in text: decision = "ignore"
            else:                 decision = "wait_for_more_signals"
        return RumorAction(type="make_decision", decision=decision)

    for char in CHARACTERS:
        if char in text or char.replace("_", " ") in text:
            return RumorAction(type="message_character", target=char, content=text)

    if any(w in text for w in ["reddit", "post", "forum"]):
        return RumorAction(type="post_reddit", content=text)

    return RumorAction(type="wait")


def _action_label(action):
    s = action.type
    if action.target:
        s += f" -> {action.target}"
    if action.decision:
        s += f" -> {action.decision}"
    return s


def _action_tiny(action):
    if action.type == "wait":
        return "wait"
    if action.type == "message_character" and action.target:
        return f"msg:{action.target}"
    if action.type == "make_decision" and action.decision:
        d = action.decision
        if d == "warn_team_quietly":
            return "dec:warn"
        if d == "request_budget_freeze":
            return "dec:freeze"
        if d == "escalate_to_leadership":
            return "dec:escalate"
        if d == "wait_for_more_signals":
            return "dec:waitinfo"
        if d == "ignore":
            return "dec:ignore"
        return "dec:" + d[:12]
    return action.type[:10]


def run_episode(env, episode_num, total_eps):
    agent_new_episode()
    obs = env.reset()
    stats = get_agent_stats()
    total_reward = 0.0
    tiny_steps = []

    if VERBOSE:
        print(f"\n{'='*48}")
        print(f"EPISODE {episode_num}/{total_eps}  epsilon={stats['epsilon']:.2f}")
        print(f"{'='*48}")
        for m in obs.messages:
            print(f"  {m}")
        for p in obs.reddit_posts:
            print(f"  reddit: {p}")

    for step in range(5):
        prompt = build_prompt(obs)
        raw = agent_generate(prompt)
        first = [l.strip() for l in raw.strip().split("\n") if l.strip()]
        display = first[0] if first else "(empty)"
        action = parse_action(raw)
        obs = env.step(action)
        total_reward += obs.reward
        label = _action_label(action)
        tiny_steps.append(_action_tiny(action))

        if VERBOSE:
            print(f"\n  step {step}  day {obs.day - 1}/5")
            print(f"    out: {display}")
            print(f"    ->  {label}")
            print(f"    r {obs.reward:+.3f}  sum {total_reward:+.3f}")
            if obs.dm_response:
                print(f"    DM: {obs.dm_response}")
            for m in obs.messages:
                print(f"    + {m}")

        if obs.done:
            gt = obs.ground_truth_revealed or {}
            event = gt.get("event_type") or gt.get("event", "?")
            rb = obs.reward_breakdown or {}
            correct = rb.get("correct_decision", False)
            ok = "yes" if correct else "no"

            if not VERBOSE:
                line = (
                    f"  {episode_num:>2}/{total_eps}  {event:<22}  "
                    f"reward {total_reward:>+7.2f}  ok {ok}"
                )
                if SHOW_STEPS:
                    line += "\n      " + " -> ".join(tiny_steps)
                print(line)

            if VERBOSE:
                truth = gt.get("core_truth") or gt.get("truth", {})
                print(f"\n  scenario: {event}")
                print(f"  truth: {truth}")
                print(f"  total {total_reward:+.3f}  correct {ok}")

            return {
                "episode": episode_num,
                "event": event,
                "reward": total_reward,
                "correct": correct,
                "accuracy": rb.get("accuracy", 0),
                "epistemic": rb.get("epistemic", 0),
                "social": rb.get("social", 0),
                "sources": len(rb.get("sources_consulted", [])),
            }

    if not VERBOSE:
        print(f"  {episode_num:>2}/{total_eps}  (no terminal state)  reward {total_reward:>+7.2f}")
    return {
        "episode": episode_num, "event": "timeout",
        "reward": total_reward, "correct": False,
        "accuracy": 0, "epistemic": 0, "social": 0, "sources": 0,
    }


def print_summary(results):
    n = len(results)
    ok = sum(1 for r in results if r["correct"])
    total_r = sum(r["reward"] for r in results)
    stats = get_agent_stats()

    print()
    print(f"  {ok}/{n} episodes correct ({ok/n:.0%})   total reward {total_r:+.2f}")
    print(f"  end epsilon {stats['epsilon']:.2f}   explore share {stats['explore_rate']:.0%}")
    print()
    for r in results:
        mark = "y" if r["correct"] else " "
        print(f"    {r['episode']:>2}  {r['event']:<22}  {r['reward']:>+7.2f}  [{mark}]")


if __name__ == "__main__":
    if not VERBOSE:
        print(
            "Rumour Mill demo (compact). "
            "RUMOUR_VERBOSE=1 full logs | RUMOUR_STEPS=1 action trace | RUMOUR_AGENT_LOG=1 agent trace\n"
        )

    env = RumorMillEnv()
    results = []
    for ep in range(1, NUM_EPISODES + 1):
        results.append(run_episode(env, ep, NUM_EPISODES))

    print_summary(results)
