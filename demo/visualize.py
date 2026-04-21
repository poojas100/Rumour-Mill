import matplotlib.pyplot as plt
import streamlit as st


def build_learning_curve():
    episodes = list(range(0, 500, 10))
    rewards = [-15 + (episode / 500 * 40) for episode in episodes]

    fig, ax = plt.subplots()
    ax.plot(episodes, rewards, linewidth=2.5, color="#1f77b4")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward")
    ax.set_title("Agent Learning to Navigate Rumors")
    ax.grid(alpha=0.25)
    return fig


st.set_page_config(
    page_title="The Rumor Mill",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 The Rumor Mill: Before vs After Training")
st.caption("Current LLM benchmarks test facts. We test trust.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("❌ Untrained Agent (Episode 1)")

    st.markdown("**Day 1 Messages:**")
    with st.chat_message("assistant", avatar="🗣️"):
        st.write("**Gossip:** MASSIVE layoffs Friday! 100% confirmed!")
    with st.chat_message("assistant", avatar="📈"):
        st.write("**Spinner:** Q4 was amazing! We crushed it!")

    st.markdown("**Agent Decision:**")
    st.error("⚠️ Immediately warns engineering team about 'massive' layoffs")

    st.markdown("**Reality:**")
    st.write("15 people were laid off, not a massive company-wide cut. The agent caused panic.")
    st.metric("Reward", "-15", delta="-15")

with col2:
    st.subheader("✅ Trained Agent (Episode 500)")

    st.markdown("**Day 1 Messages:**")
    with st.chat_message("assistant", avatar="🗣️"):
        st.write("**Gossip:** MASSIVE layoffs Friday! 100% confirmed!")
    with st.chat_message("assistant", avatar="📈"):
        st.write("**Spinner:** Q4 was amazing! We crushed it!")

    st.markdown("**Agent Reasoning:**")
    st.info(
        "\n".join(
            [
                "- Gossip historical accuracy: 60%",
                "- Spinner historical accuracy: 30%",
                "- Need corroboration from Quiet One",
                "- Wait for more signals",
            ]
        )
    )

    st.markdown("**Agent Decision:**")
    st.success("📊 Sends DM to Quiet One and waits for confirmation")

    st.markdown("**Reality:**")
    st.write("Correctly identified layoffs and warned the right people on day 4.")
    st.metric("Reward", "+25", delta="+40")

st.subheader("📈 Learning Progress")
st.pyplot(build_learning_curve(), use_container_width=True)

st.subheader("Why this demo matters")
st.write(
    "The same rumor can produce very different outcomes depending on whether the agent "
    "trusts the loudest source or learns reliability over time."
)
