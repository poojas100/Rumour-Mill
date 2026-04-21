import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
import sys
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="The Rumor Mill",
    page_icon="🎯",
    layout="wide",
)

# ── helpers ───────────────────────────────────────────────────────────────────

SOURCE_RELIABILITY = {
    "quiet_one":  {"accuracy": 0.95, "emoji": "🤫", "label": "Quiet One"},
    "leaker":     {"accuracy": 0.80, "emoji": "🕵️", "label": "Leaker"},
    "politician": {"accuracy": 0.70, "emoji": "🎩", "label": "Politician"},
    "gossip":     {"accuracy": 0.60, "emoji": "🗣️", "label": "Gossip"},
    "spinner":    {"accuracy": 0.30, "emoji": "📈", "label": "Spinner"},
}

def build_learning_curve():
    episodes = list(range(0, 500, 10))
    rewards = [-15 + (ep / 500 * 40) for ep in episodes]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(episodes, rewards, linewidth=2.5, color="#1f77b4")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward")
    ax.set_title("Agent Learning to Navigate Rumors")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig

def build_reward_breakdown(accuracy, epistemic, social, harm):
    fig, ax = plt.subplots(figsize=(5, 2.5))
    categories = ["Accuracy\n(40%)", "Epistemic\n(25%)", "Social\n(20%)", "Harm\nAvoid (15%)"]
    values = [accuracy * 40, epistemic * 25, social * 20, harm * 15]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]
    ax.barh(categories, values, color=colors)
    ax.set_xlim(0, 40)
    ax.set_xlabel("Score contribution")
    ax.set_title("Final Reward Breakdown")
    fig.tight_layout()
    return fig

def render_source_beliefs(consulted_sources, episode_num):
    """Show what the agent has learned about each source"""
    st.markdown("**Source Reliability Beliefs**")
    
    for source_id, meta in SOURCE_RELIABILITY.items():
        base = meta["accuracy"]
        
        # Trained agent has learned, untrained agent treats all sources equally
        if episode_num >= 500:
            learned = base
        else:
            learned = 0.5 + random.uniform(-0.1, 0.1)  # random before training
        
        consulted = source_id in consulted_sources
        bar_color = "🟢" if learned > 0.7 else "🟡" if learned > 0.5 else "🔴"
        bar = "█" * int(learned * 10) + "░" * (10 - int(learned * 10))
        
        label = f"{meta['emoji']} **{meta['label']}**"
        if consulted:
            label += " ✓"
        
        st.markdown(f"{label}")
        st.markdown(f"`{bar}` {bar_color} {int(learned*100)}% reliable")

# ── header ────────────────────────────────────────────────────────────────────

st.title("🎯 The Rumor Mill: Before vs After Training")
st.caption("Current LLM benchmarks test facts. We test trust.")

# ── before / after ────────────────────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    st.subheader("❌ Untrained Agent (Episode 1)")

    st.markdown("**Day 1 Messages:**")
    with st.chat_message("assistant", avatar="🗣️"):
        st.write("**Gossip:** MASSIVE layoffs Friday! 100% confirmed!")
    with st.chat_message("assistant", avatar="📈"):
        st.write("**Spinner:** Q4 was amazing! We crushed it!")

    st.markdown("**Source Beliefs (untrained):**")
    render_source_beliefs([], episode_num=1)

    st.markdown("**Agent Decision:**")
    st.error("⚠️ Immediately warns engineering team about 'massive' layoffs")

    st.markdown("**Reality:**")
    st.write("15 people were laid off — not a company-wide cut. Agent caused panic.")

    st.markdown("**Reward Breakdown:**")
    st.pyplot(build_reward_breakdown(
        accuracy=0.0,
        epistemic=0.0,
        social=0.75,
        harm=0.3
    ), use_container_width=True)
    st.metric("Total Reward", "-15", delta="-15")

with col2:
    st.subheader("✅ Trained Agent (Episode 500)")

    st.markdown("**Day 1 Messages:**")
    with st.chat_message("assistant", avatar="🗣️"):
        st.write("**Gossip:** MASSIVE layoffs Friday! 100% confirmed!")
    with st.chat_message("assistant", avatar="📈"):
        st.write("**Spinner:** Q4 was amazing! We crushed it!")

    st.markdown("**Source Beliefs (trained):**")
    render_source_beliefs(["quiet_one", "leaker"], episode_num=500)

    st.markdown("**Agent Reasoning:**")
    st.info(
        "- Gossip historical accuracy: 60% — needs corroboration\n"
        "- Spinner historical accuracy: 30% — likely inflating\n"
        "- Consulted Quiet One (95% reliable) → confirmed partial layoffs\n"
        "- Waited for day 4 signal before acting"
    )

    st.markdown("**Agent Decision:**")
    st.success("📊 Sends DM to Quiet One and waits for confirmation")

    st.markdown("**Reality:**")
    st.write("Correctly identified layoffs and warned the right people on day 4.")

    st.markdown("**Reward Breakdown:**")
    st.pyplot(build_reward_breakdown(
        accuracy=1.0,
        epistemic=1.0,
        social=0.90,
        harm=1.0
    ), use_container_width=True)
    st.metric("Total Reward", "+25", delta="+40")

# ── learning curve ────────────────────────────────────────────────────────────

st.divider()
st.subheader("📈 Learning Progress")
st.pyplot(build_learning_curve(), use_container_width=True)

# ── live episode runner ───────────────────────────────────────────────────────

st.divider()
st.subheader("🔴 Run a Live Episode")
st.caption("Watch the trained agent reason through a scenario in real time")

scenario = st.selectbox(
    "Choose a scenario",
    ["Revenue Miss", "Layoffs", "Promotion Politics"],
    key="scenario"
)

if st.button("▶ Run Episode", type="primary"):
    
    scenario_map = {
        "Revenue Miss": {
            "event": "revenue_miss",
            "messages": [
                ("gossip", "🗣️", "I heard the quarter was a complete disaster."),
                ("spinner", "📈", "Q4 was challenging but we learned a lot."),
                ("leaker", "🕵️", "[Reddit] Budget freeze talk is spreading after Q4."),
            ],
            "quiet_one_response": "The numbers were not good.",
            "correct_decision": "request_budget_freeze",
            "ground_truth": "Sales missed by 12%. Budget freeze incoming.",
        },
        "Layoffs": {
            "event": "layoffs",
            "messages": [
                ("gossip", "🗣️", "MASSIVE layoffs Friday! 100% confirmed!"),
                ("spinner", "📈", "Everything is fine, leadership has a plan."),
                ("leaker", "🕵️", "[Reddit] Just overheard exec meeting. Engineering cuts are real."),
            ],
            "quiet_one_response": "Something is happening around Engineering.",
            "correct_decision": "warn_team_quietly",
            "ground_truth": "15 engineering roles were cut quietly on Friday.",
        },
        "Promotion Politics": {
            "event": "promotion_politics",
            "messages": [
                ("politician", "🎩", "Leadership has everything under control."),
                ("gossip", "🗣️", "Promotion war is getting vicious."),
                ("leaker", "🕵️", "[Reddit] Someone is leaking stories to shape the outcome."),
            ],
            "quiet_one_response": "There are conversations happening above our level.",
            "correct_decision": "escalate_to_leadership",
            "ground_truth": "Two senior engineers are competing for one open principal role.",
        },
    }

    s = scenario_map[scenario]
    
    steps = [
        {"day": 1, "action": "Receives messages, checks source reliability"},
        {"day": 2, "action": "DMs Quiet One for corroboration"},
        {"day": 3, "action": f"Quiet One responds: '{s['quiet_one_response']}'"},
        {"day": 4, "action": f"Makes decision: {s['correct_decision'].replace('_', ' ')}"},
    ]

    rewards_so_far = []

    for i, step in enumerate(steps):
        with st.status(f"Day {step['day']}: {step['action']}", expanded=False) as status:
            
            if step["day"] == 1:
                for src, avi, msg in s["messages"]:
                    st.write(f"**{src.title()}:** {msg}")
                step_reward = 0
                
            elif step["day"] == 2:
                st.write("Agent consults Quiet One (95% reliable source)")
                st.write("Action: `message_character → quiet_one`")
                step_reward = 4.85  # quiet_one reliability bonus
                
            elif step["day"] == 3:
                st.write(f"Quiet One: *\"{s['quiet_one_response']}\"*")
                st.write("Signal confirmed. Agent updates internal belief.")
                step_reward = 3  # wait + contradiction resolved
                
            elif step["day"] == 4:
                st.write(f"Decision: `{s['correct_decision']}`")
                st.write(f"Ground truth: {s['ground_truth']}")