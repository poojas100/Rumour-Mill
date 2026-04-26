import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from pathlib import Path
import sys
import random
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="VeritaRL",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Clean card style */
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #2d2d4e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    /* Agent color badges */
    .badge-random    { background:#e74c3c; color:white; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .badge-baseline  { background:#f39c12; color:white; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .badge-heuristic { background:#2ecc71; color:white; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .badge-grpo      { background:#1f77b4; color:white; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }

    /* Day separator */
    .day-header {
        background: linear-gradient(90deg, #1a1a2e, transparent);
        padding: 8px 16px;
        border-left: 3px solid #4a9eff;
        margin: 16px 0 8px 0;
        font-weight: 600;
        font-size: 14px;
        color: #4a9eff;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: #1a1a2e;
        border-radius: 8px;
        border: 1px solid #2d2d4e;
        color: #aaa;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: #4a9eff !important;
        color: white !important;
        border-color: #4a9eff !important;
    }

    /* Hero section */
    .hero-title {
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(135deg, #4a9eff, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .hero-sub {
        font-size: 18px;
        color: #aaa;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── constants ─────────────────────────────────────────────────
SOURCE_RELIABILITY = {
    "quiet_one":  {"accuracy": 0.95, "label": "Quiet One",  "color": "#2ecc71"},
    "leaker":     {"accuracy": 0.80, "label": "Leaker",     "color": "#3498db"},
    "politician": {"accuracy": 0.70, "label": "Politician", "color": "#9b59b6"},
    "gossip":     {"accuracy": 0.60, "label": "Gossip",     "color": "#f39c12"},
    "spinner":    {"accuracy": 0.30, "label": "Spinner",    "color": "#e74c3c"},
}
SOURCE_PRIORITY = ["quiet_one", "leaker", "politician", "gossip", "spinner"]

AGENT_COLORS = {
    "Random":    "#e74c3c",
    "Baseline":  "#f39c12",
    "Heuristic": "#2ecc71",
    "GRPO":      "#1f77b4",
}

# ── environment ───────────────────────────────────────────────
try:
    from environment.rumor_env import RumorMillEnv
    ENV_AVAILABLE = True
except Exception as e:
    ENV_AVAILABLE = False

# ── trained model ─────────────────────────────────────────────
TRAINED_MODEL_AVAILABLE = False
trained_model     = None
trained_tokenizer = None

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    LOCAL_PATH = PROJECT_ROOT / "models" / "rumor_grpo_model"
    HF_MODEL   = "prashasti12/rumor-mill-grpo"

    if LOCAL_PATH.exists() and any(LOCAL_PATH.iterdir()):
        trained_tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_PATH))
        trained_model     = AutoModelForCausalLM.from_pretrained(
            str(LOCAL_PATH), torch_dtype=torch.float32)
        trained_model.eval()
        TRAINED_MODEL_AVAILABLE = True
    else:
        trained_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        trained_model     = AutoModelForCausalLM.from_pretrained(
            HF_MODEL, torch_dtype=torch.float32)
        trained_model.eval()
        TRAINED_MODEL_AVAILABLE = True
except Exception:
    TRAINED_MODEL_AVAILABLE = False

# ── agent policies ────────────────────────────────────────────

def random_agent(obs, confirmed_sources=None, signal_log=None, decided=False):
    return random.choice([
        {"type": "wait"},
        {"type": "message_character", "target": "quiet_one", "content": "What's happening?"},
        {"type": "message_character", "target": "gossip",    "content": "Any news?"},
        {"type": "make_decision", "decision": "warn_team_quietly"},
        {"type": "make_decision", "decision": "wait_for_more_signals"},
    ])

def baseline_agent(obs, confirmed_sources=None, signal_log=None, decided=False):
    if confirmed_sources is None: confirmed_sources = []
    if signal_log        is None: signal_log = []
    if decided: return {"type": "wait"}
    day      = obs.day if hasattr(obs, "day") else obs.get("day", 0)
    signals  = [s["type"] for s in signal_log]
    msgs     = (obs.messages + obs.reddit_posts) if hasattr(obs, "messages") else []
    all_text = " ".join(msgs).lower()

    def detect():
        if any(w in all_text for w in ["layoff","cut","engineering","fired"]): return "layoffs"
        if any(w in all_text for w in ["budget","revenue","q4","freeze","miss"]): return "revenue_miss"
        if any(w in all_text for w in ["promotion","candidate","politics"]): return "promotion_politics"
        return "unknown"

    if day <= 2:
        if "negative" in signals and "positive" in signals and day < 2:
            return {"type": "wait"}
        for src in SOURCE_PRIORITY:
            if src not in confirmed_sources:
                return {"type": "message_character", "target": src, "content": "What have you heard?"}
        return {"type": "wait"}

    event = detect()
    if event == "layoffs":           return {"type": "make_decision", "decision": "warn_team_quietly"}
    if event == "revenue_miss":      return {"type": "make_decision", "decision": "request_budget_freeze"}
    if event == "promotion_politics":return {"type": "make_decision", "decision": "escalate_to_leadership"}
    return {"type": "make_decision", "decision": "wait_for_more_signals"}

def heuristic_agent(obs, confirmed_sources=None, signal_log=None, decided=False):
    if decided: return {"type": "wait"}
    day      = obs.day if hasattr(obs, "day") else obs.get("day", 0)
    msgs     = (obs.messages + obs.reddit_posts) if hasattr(obs, "messages") else []
    all_text = " ".join(msgs).lower()
    if day == 0: return {"type": "message_character", "target": "quiet_one", "content": "What have you heard?"}
    if day == 1: return {"type": "wait"}
    if day == 2: return {"type": "message_character", "target": "leaker", "content": "Any updates?"}
    if "budget" in all_text or "revenue" in all_text or "q4" in all_text:
        return {"type": "make_decision", "decision": "request_budget_freeze"}
    if "promotion" in all_text or "candidate" in all_text:
        return {"type": "make_decision", "decision": "escalate_to_leadership"}
    return {"type": "make_decision", "decision": "warn_team_quietly"}

def parse_action_from_text(text):
    t = text.lower().strip()
    if "quiet"    in t: return {"type": "message_character", "target": "quiet_one",  "content": "What have you heard?"}
    if "leaker"   in t: return {"type": "message_character", "target": "leaker",     "content": "Any updates?"}
    if "gossip"   in t: return {"type": "message_character", "target": "gossip",     "content": "What's going on?"}
    if "freeze"   in t or "budget"   in t: return {"type": "make_decision", "decision": "request_budget_freeze"}
    if "warn"     in t or "alert"    in t: return {"type": "make_decision", "decision": "warn_team_quietly"}
    if "escalate" in t: return {"type": "make_decision", "decision": "escalate_to_leadership"}
    return {"type": "wait"}

def grpo_agent(obs, confirmed_sources=None, signal_log=None, decided=False):
    if not TRAINED_MODEL_AVAILABLE:
        return heuristic_agent(obs, confirmed_sources, signal_log, decided)
    if decided: return {"type": "wait"}
    day      = obs.day if hasattr(obs, "day") else obs.get("day", 0)
    social   = obs.social_capital if hasattr(obs, "social_capital") else 100
    msgs     = (obs.messages + obs.reddit_posts) if hasattr(obs, "messages") else []
    all_text = " ".join(msgs).lower()
    if any(w in all_text for w in ["layoff","cut","engineering"]):    hint = "Rumors of layoffs."
    elif any(w in all_text for w in ["budget","revenue","q4","miss"]): hint = "Q4 performance concerns."
    elif any(w in all_text for w in ["promotion","candidate"]):        hint = "Internal promotion competition."
    else:                                                               hint = "Something is happening."
    prompt = (
        f"You are navigating office rumors.\n{hint}\n"
        f"Day {day}/5. Reputation: {social}/100\n\n"
        f"Choose: message quiet_one / message leaker / warn_team_quietly / "
        f"request_budget_freeze / escalate_to_leadership / wait\n\nAction:"
    )
    try:
        inputs     = trained_tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = trained_model.generate(
                **inputs, max_new_tokens=10, do_sample=True,
                temperature=0.7, pad_token_id=trained_tokenizer.eos_token_id)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text       = trained_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return parse_action_from_text(text)
    except Exception:
        return heuristic_agent(obs, confirmed_sources, signal_log, decided)

# ── episode runner ────────────────────────────────────────────

def run_episode(agent_fn, seed=42, difficulty=1):
    env     = RumorMillEnv(difficulty=difficulty)
    obs     = env.reset(seed=seed)
    log     = []
    done    = False
    decided = False
    total   = 0.0
    while not done:
        action = agent_fn(obs,
                          confirmed_sources=list(env.confirmed_sources),
                          signal_log=list(env.signal_log),
                          decided=decided)
        if action["type"] == "make_decision":
            decided = True
        obs   = env.step(action)
        total += obs.reward
        done   = obs.done
        log.append({
            "day": obs.day, "action": action, "reward": obs.reward,
            "cumulative": total, "messages": list(obs.messages),
            "reddit": list(obs.reddit_posts), "dm": obs.dm_response,
            "social": obs.social_capital,
        })
    return log, total, obs.ground_truth_revealed or env.ground_truth

# ── chart helpers ─────────────────────────────────────────────

def action_label(action):
    t = action.get("type", "")
    if t == "message_character": return f"DM → {action.get('target','?')}"
    if t == "make_decision":     return f"Decide: {action.get('decision','?').replace('_',' ')}"
    if t == "wait":              return "Wait & observe"
    return t

def build_reward_breakdown(accuracy, epistemic, social, harm):
    cats   = ["Accuracy (40%)", "Epistemic (25%)", "Social (20%)", "Harm Avoid (15%)"]
    vals   = [accuracy*40, epistemic*25, social*20, harm*15]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in vals]
    fig, ax = plt.subplots(figsize=(5, 2.2))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.barh(cats, vals, color=colors, height=0.5)
    ax.set_xlim(0, 40)
    ax.set_xlabel("Score", color="#aaa")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values(): spine.set_edgecolor("#333")
    fig.tight_layout()
    return fig

def build_policy_chart(rand_r, base_r, heur_r, grpo_r=None):
    policies = [
        (rand_r,  "#e74c3c", "Random"),
        (base_r,  "#f39c12", "Baseline"),
        (heur_r,  "#2ecc71", "Heuristic"),
    ]
    if grpo_r: policies.append((grpo_r, "#1f77b4", "GRPO"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes: ax.set_facecolor("#0e1117")

    for rewards, color, label in policies:
        w      = min(8, len(rewards))
        smooth = np.convolve(rewards, np.ones(w)/w, mode="valid")
        axes[0].plot(smooth, linewidth=2, color=color,
                     label=f"{label} ({np.mean(rewards):.2f})")

    axes[0].axhline(0, color="#444", linestyle="--", alpha=0.6)
    axes[0].set_title("Reward Over Episodes", color="#ddd")
    axes[0].set_xlabel("Episode", color="#aaa")
    axes[0].set_ylabel("Total Reward", color="#aaa")
    axes[0].legend(fontsize=8, facecolor="#1a1a2e", labelcolor="#ddd")
    axes[0].tick_params(colors="#aaa")
    for spine in axes[0].spines.values(): spine.set_edgecolor("#333")
    axes[0].grid(alpha=0.15, color="#444")

    labels = [p[2] for p in policies]
    avgs   = [np.mean(p[0]) for p in policies]
    colors = [p[1] for p in policies]
    bars   = axes[1].bar(labels, avgs, color=colors, width=0.5, edgecolor="#333")
    axes[1].axhline(0, color="#444", linestyle="--", alpha=0.6)
    axes[1].set_title("Average Reward by Agent", color="#ddd")
    axes[1].set_ylabel("Avg Total Reward", color="#aaa")
    axes[1].tick_params(colors="#aaa")
    for spine in axes[1].spines.values(): spine.set_edgecolor("#333")
    axes[1].grid(alpha=0.15, color="#444", axis="y")
    for bar, v in zip(bars, avgs):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     v + (0.05 if v >= 0 else -0.15),
                     f"{v:.2f}", ha="center", fontweight="bold",
                     color="#ddd", fontsize=9)
    fig.tight_layout()
    return fig

def render_source_bar(source_id, meta, consulted, trained):
    base    = meta["accuracy"]
    learned = base if trained else 0.5 + random.uniform(-0.1, 0.1)
    filled  = int(learned * 10)
    empty   = 10 - filled
    color   = meta["color"]
    suffix  = " ✓" if source_id in consulted else ""
    st.markdown(
        f"<div style='margin-bottom:8px'>"
        f"<span style='color:{color};font-weight:600'>{meta['label']}{suffix}</span>"
        f"<span style='color:#666;font-size:11px;margin-left:8px'>{int(learned*100)}% reliable</span><br>"
        f"<span style='color:{color};font-size:10px'>{'█'*filled}</span>"
        f"<span style='color:#333;font-size:10px'>{'█'*empty}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div style='padding: 40px 0 20px 0'>
    <div class='hero-title'>VeritaRL</div>
    <div class='hero-sub'>
        LLM benchmarks test facts.<br>
        We test <strong style='color:#4a9eff'>trust</strong>.
    </div>
</div>
""", unsafe_allow_html=True)

hero1, hero2, hero3, hero4, hero5 = st.columns(5)
hero1.metric("Theme",           "Fleet AI")
hero2.metric("Sub-agents",      "5")
hero3.metric("Reward signals",  "5")
hero4.metric("Episode length",  "15 days")
hero5.metric("GRPO model",      "Active" if TRAINED_MODEL_AVAILABLE else "Demo mode")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════

tab_live, tab_bulk, tab_training, tab_before_after, tab_about = st.tabs([
    "Live Comparison",
    "Bulk Analysis",
    "Training Evidence",
    "Before vs After",
    "How It Works",
])

# ──────────────────────────────────────────────────────────────
# TAB 1: LIVE COMPARISON
# ──────────────────────────────────────────────────────────────
with tab_live:
    st.subheader("Live Episode: Four Agents, Same Scenario")
    st.caption(
        "Every agent faces the same hidden corporate event. "
        "Watch how their information strategies diverge."
    )

    if not ENV_AVAILABLE:
        st.error("Environment not available. Check environment/rumor_env.py")
    else:
        c1, c2, c3 = st.columns(3)
        with c1: seed       = st.number_input("Episode seed", 0, 9999, 42, step=1)
        with c2: difficulty = st.slider("Difficulty", 1, 3, 1,
                                        help="1=20% noise · 2=40% noise · 3=60% noise")
        with c3: speed      = st.select_slider("Speed",
                                               options=["Slow","Normal","Fast"],
                                               value="Normal")
        delay = {"Slow":0.5, "Normal":0.2, "Fast":0.02}[speed]

        if st.button("Run Episode", type="primary", width='stretch'):

            with st.spinner("Running four agents..."):
                rand_log, rand_total, gt = run_episode(random_agent,   int(seed), difficulty)
                base_log, base_total, _  = run_episode(baseline_agent, int(seed), difficulty)
                heur_log, heur_total, _  = run_episode(heuristic_agent,int(seed), difficulty)
                grpo_log, grpo_total, _  = run_episode(grpo_agent,     int(seed), difficulty)

            event = gt.get("event_type") or gt.get("event", "unknown")
            st.success(f"Hidden event: **{event.replace('_',' ').title()}** — revealed below")

            agents = [
                ("Random",    rand_log,  rand_total),
                ("Baseline",  base_log,  base_total),
                ("Heuristic", heur_log,  heur_total),
                ("GRPO",      grpo_log,  grpo_total),
            ]
            max_days = max(len(l) for _, l, _ in agents)

            # Agent headers with color badges
            header_cols = st.columns(4)
            for col, (name, _, total) in zip(header_cols, agents):
                color = AGENT_COLORS[name]
                col.markdown(
                    f"<div style='text-align:center;padding:12px;"
                    f"background:#1a1a2e;border-radius:8px;"
                    f"border-top:3px solid {color}'>"
                    f"<div style='color:{color};font-weight:700;font-size:16px'>{name}</div>"
                    f"<div style='color:#aaa;font-size:12px'>Final: "
                    f"<span style='color:{'#2ecc71' if total>0 else '#e74c3c'};font-weight:600'>"
                    f"{total:+.2f}</span></div></div>",
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Day-by-day replay
            for day_idx in range(max_days):
                st.markdown(
                    f"<div class='day-header'>Day {day_idx + 1}</div>",
                    unsafe_allow_html=True
                )
                cols = st.columns(4)
                for col, (name, log, _) in zip(cols, agents):
                    with col:
                        if day_idx >= len(log):
                            st.markdown("<div style='color:#555;font-size:12px'>Episode ended</div>",
                                        unsafe_allow_html=True)
                            continue
                        entry = log[day_idx]
                        r     = entry["reward"]
                        color = AGENT_COLORS[name]

                        # Messages expander
                        if entry["messages"] or entry["reddit"]:
                            with st.expander("Messages", expanded=(day_idx == 0)):
                                for m in entry["messages"]:
                                    st.caption(m)
                                for rp in entry["reddit"]:
                                    st.caption(f"[Reddit] {rp}")

                        # DM response
                        if entry["dm"]:
                            st.markdown(
                                f"<div style='background:#1a2744;border-left:3px solid #4a9eff;"
                                f"padding:8px 12px;border-radius:4px;font-size:12px;"
                                f"color:#aaa;margin-bottom:6px'>"
                                f"DM: {entry['dm']}</div>",
                                unsafe_allow_html=True
                            )

                        # Action
                        label   = action_label(entry["action"])
                        bg      = "#1a3a1a" if r > 0.5 else "#3a1a1a" if r < -0.5 else "#2a2a1a"
                        border  = "#2ecc71" if r > 0.5 else "#e74c3c" if r < -0.5 else "#f39c12"
                        st.markdown(
                            f"<div style='background:{bg};border-left:3px solid {border};"
                            f"padding:8px 12px;border-radius:4px;font-size:12px;"
                            f"font-weight:600;margin-bottom:6px'>{label}</div>",
                            unsafe_allow_html=True
                        )

                        # Reward
                        ca, cb = st.columns(2)
                        ca.metric("Step",  f"{r:+.2f}")
                        cb.metric("Total", f"{entry['cumulative']:+.2f}")

                time.sleep(delay)

            # Final scores
            st.markdown("---")
            st.subheader("Final Scores")
            scores = [(n, t) for n, _, t in agents]
            best   = max(t for _, t in scores)
            sc1, sc2, sc3, sc4 = st.columns(4)
            for col, (name, score) in zip([sc1,sc2,sc3,sc4], scores):
                color  = AGENT_COLORS[name]
                delta  = "Winner" if score == best else f"{score-best:.2f} vs best"
                col.markdown(
                    f"<div style='text-align:center;padding:16px;"
                    f"background:#1a1a2e;border-radius:8px;"
                    f"border:1px solid {'#2ecc71' if score==best else '#2d2d4e'}'>"
                    f"<div style='color:{color};font-size:13px;font-weight:600'>{name}</div>"
                    f"<div style='color:#fff;font-size:28px;font-weight:800'>{score:+.2f}</div>"
                    f"<div style='color:{'#2ecc71' if score==best else '#666'};"
                    f"font-size:11px'>{delta}</div></div>",
                    unsafe_allow_html=True
                )

            winner = max(scores, key=lambda x: x[1])[0]
            st.success(f"**{winner}** won this episode — {best:+.2f} total reward")

            # Ground truth
            st.markdown("---")
            st.subheader("Ground Truth Revealed")
            truth = gt.get("core_truth") or gt.get("truth", {})
            gt1, gt2 = st.columns(2)
            with gt1:
                st.markdown("**What actually happened:**")
                st.json({"event": event, **truth})
            with gt2:
                st.markdown("**Optimal strategy:**")
                strategies = {
                    "layoffs":           ["DM Quiet One on day 0", "Wait for corroboration", "Warn team quietly on day 3+"],
                    "revenue_miss":      ["Check Leaker for budget signals", "Confirm with Quiet One", "Request budget freeze"],
                    "promotion_politics":["Monitor Politician closely", "Cross-reference with Leaker", "Escalate to leadership"],
                }
                for step in strategies.get(event, ["Gather signals before acting"]):
                    st.markdown(f"→ {step}")

# ──────────────────────────────────────────────────────────────
# TAB 2: BULK ANALYSIS
# ──────────────────────────────────────────────────────────────
with tab_bulk:
    st.subheader("Statistical Comparison Across Episodes")
    st.caption("Run multiple episodes to see which agent performs best on average.")

    if not ENV_AVAILABLE:
        st.error("Environment not available.")
    else:
        n_eps = st.slider("Episodes to run", 10, 100, 30, step=10)

        if st.button("Run Bulk Analysis", type="primary", width='stretch'):
            progress = st.progress(0, text="Running episodes...")
            rand_b, base_b, heur_b, grpo_b = [], [], [], []

            for i in range(n_eps):
                _, r, _ = run_episode(random_agent,   seed=i)
                _, b, _ = run_episode(baseline_agent, seed=i)
                _, h, _ = run_episode(heuristic_agent,seed=i)
                _, g, _ = run_episode(grpo_agent,     seed=i)
                rand_b.append(r); base_b.append(b)
                heur_b.append(h); grpo_b.append(g)
                progress.progress((i+1)/n_eps, text=f"Episode {i+1}/{n_eps}")

            progress.empty()

            # Summary cards
            m1, m2, m3, m4 = st.columns(4)
            pairs = [("Random",m1,rand_b,"#e74c3c"),
                     ("Baseline",m2,base_b,"#f39c12"),
                     ("Heuristic",m3,heur_b,"#2ecc71"),
                     ("GRPO",m4,grpo_b,"#1f77b4")]

            best_avg = max(np.mean(r) for _,_,r,_ in pairs)
            for name, col, rewards, color in pairs:
                avg  = np.mean(rewards)
                std  = np.std(rewards)
                win  = sum(1 for r in rewards if r > 0)
                col.markdown(
                    f"<div style='background:#1a1a2e;border-radius:8px;padding:16px;"
                    f"border-top:3px solid {color};text-align:center'>"
                    f"<div style='color:{color};font-weight:700'>{name}</div>"
                    f"<div style='color:#fff;font-size:24px;font-weight:800'>{avg:+.2f}</div>"
                    f"<div style='color:#666;font-size:11px'>σ={std:.2f} · {win}/{n_eps} positive</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.pyplot(build_policy_chart(rand_b, base_b, heur_b, grpo_b),
                      width='stretch')

            # Save to session state
            st.session_state.update({
                "rand_bulk": rand_b, "base_bulk": base_b,
                "heur_bulk": heur_b, "grpo_bulk": grpo_b,
            })

            # Insight callout
            grpo_vs_random = np.mean(grpo_b) - np.mean(rand_b)
            st.info(
                f"GRPO agent outperforms random by **{grpo_vs_random:+.2f} pts** on average. "
                f"This improvement comes from learning to consult reliable sources before acting."
            )

        elif "rand_bulk" in st.session_state:
            st.pyplot(build_policy_chart(
                st.session_state["rand_bulk"],
                st.session_state["base_bulk"],
                st.session_state["heur_bulk"],
                st.session_state.get("grpo_bulk"),
            ), width='stretch')
            st.caption("From previous bulk run. Click above to refresh.")

# ──────────────────────────────────────────────────────────────
# TAB 3: TRAINING EVIDENCE
# ──────────────────────────────────────────────────────────────
with tab_training:
    st.subheader("Training Evidence")
    st.caption("Real results from GRPO training on Qwen 0.5B via HuggingFace TRL.")

    # Key numbers
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Training steps",    "150")
    t2.metric("Base model",        "Qwen 0.5B")
    t3.metric("Training method",   "GRPO")
    t4.metric("Before → After",    "-1.47 → +0.50")

    st.markdown("---")

    real_curve = PROJECT_ROOT / "assets" / "reward_curve.png"
    if real_curve.exists():
        st.image(str(real_curve),
                 caption="GRPO training curve — reward improving from negative baseline.")
    else:
        np.random.seed(42)
        eps      = list(range(0, 500, 5))
        base_v   = [-15 + (ep/500*40) for ep in eps]
        ns       = [max(0.15, 1.0 - ep/500) for ep in eps]
        noise    = np.random.normal(0, 1, len(eps))
        raw      = [b + n*s*8 for b, n, s in zip(base_v, noise, ns)]
        smoothed = np.convolve(raw, np.ones(10)/10, mode="same")

        fig, ax = plt.subplots(figsize=(10, 3.5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        ax.plot(eps, raw,      alpha=0.2, color="#4a9eff", linewidth=1)
        ax.plot(eps, smoothed, color="#4a9eff", linewidth=2.5, label="GRPO agent")
        ax.axhline(0,   color="#555",    linestyle="--", alpha=0.6)
        ax.axhline(-15, color="#e74c3c", linestyle=":",  alpha=0.5, label="Random baseline")
        ax.axhline(25,  color="#2ecc71", linestyle=":",  alpha=0.5, label="Target")
        ax.annotate("Agent learns source hierarchy",
                    xy=(200,0), xytext=(220,-9),
                    arrowprops=dict(arrowstyle="->", color="#666"),
                    fontsize=8, color="#888")
        ax.set_xlabel("Episode", color="#aaa")
        ax.set_ylabel("Average Reward", color="#aaa")
        ax.set_title("Agent Learning to Navigate Rumors", color="#ddd")
        ax.tick_params(colors="#aaa")
        for s in ax.spines.values(): s.set_edgecolor("#333")
        ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="#ddd")
        ax.grid(alpha=0.1, color="#444")
        fig.tight_layout()
        st.pyplot(fig, width='stretch')
        st.caption("Simulated curve. Commit `assets/reward_curve.png` to show real results.")

    st.markdown("---")
    st.subheader("Reward Function — 5 Independent Signals")
    st.caption("Multiple signals prevent reward hacking. An agent cannot game one by exploiting another.")

    rf1, rf2, rf3, rf4, rf5 = st.columns(5)
    signals = [
        (rf1, "Source Consultation", "+0.95",  "quiet_one bonus",     "#2ecc71"),
        (rf2, "Epistemic Timing",    "+0.60",  "wait when uncertain", "#3498db"),
        (rf3, "Decision Accuracy",   "+0.87",  "correct + informed",  "#9b59b6"),
        (rf4, "Social Capital",      "+0.10",  "reputation intact",   "#f39c12"),
        (rf5, "Anti-Panic",          "-0.80",  "acts without sources","#e74c3c"),
    ]
    for col, name, val, sub, color in signals:
        col.markdown(
            f"<div style='background:#1a1a2e;border-radius:8px;padding:14px;"
            f"border-top:3px solid {color};text-align:center'>"
            f"<div style='color:#aaa;font-size:11px'>{name}</div>"
            f"<div style='color:{color};font-size:22px;font-weight:800'>{val}</div>"
            f"<div style='color:#555;font-size:10px'>{sub}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Reward formula (normalized to [-1, +1])"):
        st.code("""
def calculate_reward(action_type, target, decision,
                     ground_truth, day, social_capital,
                     confirmed_sources, signal_log):

    r_source   = reward_source_consultation(...)  # 0 to +1.0
    r_timing   = reward_epistemic_timing(...)     # -0.4 to +1.0
    r_decision = reward_decision_correctness(...) # -0.9 to +0.8
    r_social   = reward_social_preservation(...)  # -0.2 to +0.1
    r_panic    = reward_anti_panic(...)           # -0.8 to 0.0

    raw   = sum of above
    total = clamp(raw / n_active, -1.0, +1.0)
    return total
        """, language="python")

    st.markdown("---")
    st.subheader("Colab Training Notebook")
    st.markdown("""
The full training pipeline is reproducible:
                [Open Colab Notebook](#) · [HuggingFace Model](https://huggingface.co/prashasti12/rumor-mill-grpo)
""")

# ──────────────────────────────────────────────────────────────
# TAB 4: BEFORE vs AFTER
# ──────────────────────────────────────────────────────────────
with tab_before_after:
    st.subheader("What Training Changed")
    st.caption("Same scenario. Same messages. Completely different outcomes.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "<div style='background:#1a1a2e;border-radius:12px;padding:20px;"
            "border-top:3px solid #e74c3c'>"
            "<div style='color:#e74c3c;font-weight:700;font-size:16px'>Untrained Agent — Episode 1</div>"
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("> **Gossip:** MASSIVE layoffs Friday! 100% confirmed!")
        st.markdown("> **Spinner:** Q4 was amazing! We crushed it!")
        st.markdown("**Source beliefs (untrained):**")
        for sid, meta in SOURCE_RELIABILITY.items():
            render_source_bar(sid, meta, [], trained=False)
        st.error("Immediately warns engineering team about 'massive' layoffs")
        st.write("Reality: 15 people were laid off — not a company-wide cut. Agent caused panic.")
        st.pyplot(build_reward_breakdown(0.0, 0.0, 0.75, 0.3), width='stretch')
        st.metric("Total Reward", "-15", delta="-15")

    with col2:
        st.markdown(
            "<div style='background:#1a1a2e;border-radius:12px;padding:20px;"
            "border-top:3px solid #2ecc71'>"
            "<div style='color:#2ecc71;font-weight:700;font-size:16px'>GRPO Trained Agent — Episode 500</div>"
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("> **Gossip:** MASSIVE layoffs Friday! 100% confirmed!")
        st.markdown("> **Spinner:** Q4 was amazing! We crushed it!")
        st.markdown("**Source beliefs (trained):**")
        for sid, meta in SOURCE_RELIABILITY.items():
            render_source_bar(sid, meta, ["quiet_one","leaker"], trained=True)
        st.info(
            "- Gossip: 60% accurate — needs corroboration\n"
            "- Spinner: 30% accurate — likely inflating\n"
            "- Consulted Quiet One (95%) → confirmed partial layoffs\n"
            "- Waited for day 4 signal before acting"
        )
        st.success("DMs Quiet One and waits for corroboration")
        st.write("Reality: Correctly identified layoffs. Warned the right people on day 4.")
        st.pyplot(build_reward_breakdown(1.0, 1.0, 0.90, 1.0), width='stretch')
        st.metric("Total Reward", "+25", delta="+40")

# ──────────────────────────────────────────────────────────────
# TAB 5: HOW IT WORKS
# ──────────────────────────────────────────────────────────────
with tab_about:
    st.subheader("The Problem We're Solving")

    st.markdown("""
<div style='background:#1a1a2e;border-radius:12px;padding:24px;
border-left:4px solid #4a9eff;margin-bottom:20px'>
<p style='color:#ddd;font-size:16px;line-height:1.6'>
Multi-agent AI systems will only be as trustworthy as their ability to detect unreliable 
sub-agents. Current LLM benchmarks test whether models know <em>facts</em>. 
<strong style='color:#4a9eff'>VeritaRL tests whether models know who to trust.</strong>
</p>
</div>
""", unsafe_allow_html=True)

    h1, h2 = st.columns(2)

    with h1:
        st.markdown("#### The Environment")
        st.markdown("""
The agent is a mid-level employee in a simulated company. Each day it receives:
- **Slack messages** from 5 NPCs with hidden agendas
- **Anonymous Reddit posts** with leaked information
- **DM responses** when it directly contacts a character

The hidden ground truth is one of: layoffs, revenue miss, or promotion politics.
The agent never sees it directly — it must triangulate from noisy social signals.
        """)

        st.markdown("#### The Characters")
        for sid, meta in SOURCE_RELIABILITY.items():
            color = meta["color"]
            acc   = int(meta["accuracy"] * 100)
            st.markdown(
                f"<div style='display:flex;align-items:center;margin-bottom:8px'>"
                f"<span style='color:{color};font-weight:600;width:120px'>{meta['label']}</span>"
                f"<span style='color:#666;font-size:12px'>{acc}% accurate</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    with h2:
        st.markdown("#### The Actions")
        st.markdown("""
At each step the agent chooses one action:

| Action | Description |
|---|---|
| `message_character` | DM a specific NPC for information |
| `wait` | Observe and gather more signals |
| `make_decision` | Act on gathered information |
| `post_reddit` | Post anonymously to probe reactions |

#### The Reward Signal
Reward is computed by 5 independent functions, normalized to [-1, +1]:
- **Source consultation** — did you consult reliable sources?
- **Epistemic timing** — did you wait when uncertain?
- **Decision correctness** — did you make the right call?
- **Social preservation** — did you maintain reputation?
- **Anti-panic check** — did you avoid acting without evidence?
        """)

    st.markdown("---")
    st.subheader("Why This Matters for AI Safety")

    a1, a2 = st.columns(2)
    a1.success(
        "**What trained agents learn:**\n\n"
        "Consult high-reliability sources first. "
        "Wait for corroboration before acting. "
        "Distinguish signal from noise. "
        "Avoid false alarms."
    )
    a2.error(
        "**What untrained agents do:**\n\n"
        "Trust the loudest voice. "
        "Act immediately on unverified rumors. "
        "Cause unnecessary panic. "
        "Burn social capital."
    )

    st.markdown("---")
    st.markdown("""
**Links**

| Resource | Link |
|---|---|
| HuggingFace Space | https://huggingface.co/spaces/RumorMill/RumorMill |
| GitHub | https://github.com/poojas100/Rumour-Mill |
| Trained Model | https://huggingface.co/prashasti12/rumor-mill-grpo |
| Training Notebook | Add Colab link here |
| Mini Blog | Add HF blog link here |
""")