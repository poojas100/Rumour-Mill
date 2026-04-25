"""
demo/sample_episodes.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.rumor_env import RumorMillEnv
from environment.models import RumorAction
from demo.inference_agent import generate as agent_generate

CHARACTERS = ["spinner", "gossip", "quiet_one", "politician", "leaker"]
DECISIONS  = [
    "warn_team_quietly", "wait_for_more_signals",
    "escalate_to_leadership", "request_budget_freeze", "ignore",
]


def build_prompt(obs) -> str:
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


def parse_action(raw: str) -> RumorAction:
    lines = [l.strip().lstrip("-•*123456789. ") for l in raw.strip().split("\n")]
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


def run_episode():     
    env = RumorMillEnv()
    obs = env.reset()

    print("\n" + "="*60)
    print("RUMOUR MILL — NEW EPISODE")
    print("="*60)
    print(f"Event hidden. Day 0 of 5.")
    print(f"Messages today: {len(obs.messages)}")
    for m in obs.messages:
        print(f"  📨 {m}")
    for p in obs.reddit_posts:
        print(f"  📋 [reddit] {p}")

    total_reward = 0.0

    for step in range(5):
        print(f"\n{'─'*60}")
        print(f"STEP {step} | Day {obs.day}/5 | Reputation: {obs.social_capital:.0f}/100")

        prompt     = build_prompt(obs)
        raw_output = agent_generate(prompt)

        first_line     = [l.strip() for l in raw_output.strip().split("\n") if l.strip()]
        display_output = first_line[0] if first_line else "(empty)"
        print(f"  🤖 Model said:  '{display_output}'")

        action = parse_action(raw_output)
        action_str = action.type
        if action.target:   action_str += f" → {action.target}"
        if action.decision: action_str += f" → {action.decision}"
        print(f"  ✅ Parsed:      {action_str}")

        obs = env.step(action)
        total_reward += obs.reward

        print(f"  💰 Reward:      {obs.reward:+.4f}  (total: {total_reward:+.4f})")
        print(f"  👥 Reputation:  {obs.social_capital:.0f}/100")

        if obs.dm_response:
            print(f"  💬 DM reply:    '{obs.dm_response}'")

        if obs.messages:
            print(f"  📨 New messages:")
            for m in obs.messages:
                print(f"     {m}")

        if obs.done:
            print(f"\n{'='*60}")
            print("EPISODE COMPLETE")
            if obs.ground_truth_revealed:
                gt = obs.ground_truth_revealed
                # ✅ FIX: support both old format and new timeline format
                event = gt.get("event_type") or gt.get("event", "unknown")
                truth = gt.get("core_truth") or gt.get("truth", {})
                domain = gt.get("domain", "")
                print(f"  🌐 Domain:     {domain}")
                print(f"  🔍 Truth was:  {event}")
                print(f"  📊 Details:    {truth}")
            print(f"  🏆 Total reward: {total_reward:+.4f}")
            if obs.reward_breakdown:
                rb = obs.reward_breakdown
                print(f"  📈 Accuracy:   {rb.get('accuracy', 0):.0%}")
                print(f"  📈 Epistemic:  {rb.get('epistemic', 0):.0%}")
                print(f"  📈 Social:     {rb.get('social', 0):.0%}")
                print(f"  📈 Sources:    {rb.get('sources_consulted', [])}")
                print(f"  📈 Correct:    {rb.get('correct_decision', False)}")
            break


if __name__ == "__main__":
    run_episode()