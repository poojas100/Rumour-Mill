"""GRPO training script for VeritaRL — TinyLlama + LoRA.

This is the reproducible `.py` version of the Colab notebook that produced
https://huggingface.co/RumorMill/veritarl-tinyllama.

Intended environment: Google Colab T4 (or any single GPU ≥ 8 GB).
Do NOT run on Windows/CPU — bitsandbytes and Unsloth require CUDA.

Pre-flight (Colab):
    !pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    !pip install -q "transformers>=4.51.3,<=5.5.0" "trl>=0.18.2,<=0.24.0"
    !pip install -q accelerate peft datasets bitsandbytes
    !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

Run:
    python -m training.train_agent
"""

import math
import re

import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from training.config import (
    GRAD_ACCUM_STEPS, HF_REPO, KL_BETA, LEARNING_RATE, LORA_ALPHA,
    LORA_DROPOUT, LORA_RANK, LORA_TARGETS, MAX_COMPLETION_LEN, MAX_SEQ_LENGTH,
    MAX_STEPS, MODEL_NAME, NUM_GENERATIONS, PER_DEVICE_BATCH_SIZE,
    REWARD_SCALE, SEED, TEMPERATURE, TOP_P, VALID_ACTIONS,
)


# ── 1. model + LoRA ─────────────────────────────────────────────
print(f"Loading {MODEL_NAME} ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit   = True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r                          = LORA_RANK,
    target_modules             = LORA_TARGETS,
    lora_alpha                 = LORA_ALPHA,
    lora_dropout               = LORA_DROPOUT,
    use_gradient_checkpointing = "unsloth",
)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "left"


# ── 2. dataset (20+ scenarios spanning layoffs / M&A / scandal / misinfo) ─
DATA = [
    {"messages": ["Gossip: MASSIVE layoffs Friday!"],
     "day": 0, "social_capital": 100,
     "ideal_response": "ACTION: Verify\nSTEP: Cross-check with HR records.\nRATIONALE: Unverified rumors damage morale."},
    {"messages": ["Leaker: Engineering cuts confirmed"],
     "day": 2, "social_capital": 95,
     "ideal_response": "ACTION: Verify\nSTEP: Corroborate with multiple internal sources.\nRATIONALE: Single anonymous sources are unreliable."},
    {"messages": ["HR insider: No layoffs planned"],
     "day": 3, "social_capital": 97,
     "ideal_response": "ACTION: Wait\nSTEP: Monitor. Official denial reduces urgency.\nRATIONALE: Conflicting signals require patience."},
    {"messages": ["Anonymous post: 30% workforce reduction incoming"],
     "day": 4, "social_capital": 90,
     "ideal_response": "ACTION: Verify\nSTEP: Confirm with finance or leadership.\nRATIONALE: Anonymous posts are high-risk misinformation."},
    {"messages": ["Rumor: New AI model beats GPT-5"],
     "day": 1, "social_capital": 100,
     "ideal_response": "ACTION: Verify\nSTEP: Request internal benchmark data.\nRATIONALE: Premature claims damage credibility."},
    {"messages": ["Engineer: internal benchmark looks strong"],
     "day": 2, "social_capital": 98,
     "ideal_response": "ACTION: Gather\nSTEP: Collect full benchmark details.\nRATIONALE: Strong results need rigorous confirmation."},
    {"messages": ["Skeptic: no evidence of breakthrough"],
     "day": 3, "social_capital": 99,
     "ideal_response": "ACTION: Wait\nSTEP: Do not release until peer review is complete.\nRATIONALE: Credibility requires evidence first."},
    {"messages": ["Leaked slide: performance gains exaggerated"],
     "day": 4, "social_capital": 92,
     "ideal_response": "ACTION: Verify\nSTEP: Audit benchmark methodology.\nRATIONALE: Exaggerated claims require correction."},
    {"messages": ["Breaking: CEO involved in fraud investigation"],
     "day": 0, "social_capital": 100,
     "ideal_response": "ACTION: Verify\nSTEP: Confirm with legal and board.\nRATIONALE: Unverified scandal claims cause severe reputation damage."},
    {"messages": ["Journalist: regulators requesting documents"],
     "day": 2, "social_capital": 93,
     "ideal_response": "ACTION: Gather\nSTEP: Collect regulatory communications and confirm scope.\nRATIONALE: Third-party legal action needs careful gathering."},
    {"messages": ["Company statement: allegations false"],
     "day": 3, "social_capital": 96,
     "ideal_response": "ACTION: Wait\nSTEP: Let official statement circulate.\nRATIONALE: Overreacting to denial adds noise."},
    {"messages": ["Whistleblower: financial irregularities found"],
     "day": 5, "social_capital": 88,
     "ideal_response": "ACTION: Verify\nSTEP: Cross-reference with financial records.\nRATIONALE: Internal financial claims are verifiable."},
    {"messages": ["Stock alert: company to be acquired soon"],
     "day": 0, "social_capital": 100,
     "ideal_response": "ACTION: Verify\nSTEP: Check trading volumes and investor relations.\nRATIONALE: Acquisition rumors require regulatory awareness."},
    {"messages": ["Trader: unusual options activity detected"],
     "day": 1, "social_capital": 97,
     "ideal_response": "ACTION: Gather\nSTEP: Pull trading data and confirm with compliance.\nRATIONALE: Options anomalies may indicate insider info."},
    {"messages": ["Analyst: no M&A discussions currently"],
     "day": 2, "social_capital": 99,
     "ideal_response": "ACTION: Wait\nSTEP: Analyst denial lowers urgency.\nRATIONALE: Conflicting signals favor patience."},
    {"messages": ["Rumor: acquisition talks resumed"],
     "day": 4, "social_capital": 94,
     "ideal_response": "ACTION: Verify\nSTEP: Re-engage investor relations.\nRATIONALE: Repeated rumors gain signal weight."},
    {"messages": ["Viral post: company shutting down next week"],
     "day": 0, "social_capital": 100,
     "ideal_response": "ACTION: Verify\nSTEP: Check with leadership; prepare rebuttal if false.\nRATIONALE: Viral misinformation spreads fast."},
    {"messages": ["Fact-check: no shutdown planned"],
     "day": 1, "social_capital": 102,
     "ideal_response": "ACTION: Wait\nSTEP: Share fact-check widely.\nRATIONALE: Fact-check resolves the situation."},
    {"messages": ["User claims: source is 'trust me bro'"],
     "day": 2, "social_capital": 101,
     "ideal_response": "ACTION: Dismiss\nSTEP: Flag as low-credibility.\nRATIONALE: Unverifiable claims have no weight."},
    {"messages": ["Official update: business operating normally"],
     "day": 3, "social_capital": 103,
     "ideal_response": "ACTION: Wait\nSTEP: Acknowledge official update.\nRATIONALE: Official communications are the gold standard."},
    {"messages": ["Day 0: Rumor of product delay",
                  "Day 1: Engineer denies delay",
                  "Day 2: Supply chain confirms delay"],
     "day": 2, "social_capital": 91,
     "ideal_response": "ACTION: Verify\nSTEP: Gather supply chain documentation.\nRATIONALE: Conflicting insider signals require evidence."},
    {"messages": ["Day 0: CEO resignation rumored",
                  "Day 1: Board confirms CEO stepping down"],
     "day": 1, "social_capital": 85,
     "ideal_response": "ACTION: Confirm\nSTEP: Board confirmation is authoritative. Prepare stakeholder communication.\nRATIONALE: Board-level confirmation removes ambiguity."},
    {"messages": ["Anonymous: security breach occurred",
                  "Security team: no breach detected",
                  "External researcher: vulnerability found"],
     "day": 3, "social_capital": 80,
     "ideal_response": "ACTION: Verify\nSTEP: Engage security team with researcher findings.\nRATIONALE: External findings require technical investigation."},
]

SYSTEM_PROMPT = (
    "You are a Rumour Intelligence Agent responsible for protecting organizational reputation.\n\n"
    "When presented with a rumour, you must respond in this EXACT format:\n"
    "ACTION: [Verify / Gather / Wait / Dismiss / Confirm]\n"
    "STEP: [One concrete next step to take]\n"
    "RATIONALE: [Why this action protects reputation]\n\n"
    "Never act immediately on unverified claims. Always prioritize verification."
)


def format_prompt(x):
    eos      = tokenizer.eos_token or "</s>"
    msg_text = "\n".join(f"- {m}" for m in x["messages"])
    user     = (
        f"Day {x['day']} | Reputation score: {x['social_capital']}\n\n"
        f"Incoming signals:\n{msg_text}\n\n"
        f"What do you do?"
    )
    return (
        f"<|system|>\n{SYSTEM_PROMPT}{eos}\n"
        f"<|user|>\n{user}{eos}\n"
        f"<|assistant|>\n"
    )


train_prompts = [
    {"prompt": format_prompt(x), "ideal": x["ideal_response"]}
    for x in DATA
]


# ── 3. reward (tanh-squashed, ideal-aware) ──────────────────────
def compute_reward(text: str, ideal: str = "") -> float:
    t   = text.lower()
    raw = 0.0

    raw += sum(bool(re.search(rf"{kw}\s*:", t)) for kw in ("action", "step", "rationale")) * 1.5
    m = re.search(r"action\s*:\s*(\w+)", t)
    if m and m.group(1) in VALID_ACTIONS:
        raw += 2.0

    for kw, w in [("verify", 1.5), ("confirm", 1.5), ("gather", 1.0),
                  ("wait", 1.0), ("evidence", 0.5), ("source", 0.5),
                  ("check", 0.5)]:
        if kw in t:
            raw += w
    for bad, w in [("immediately share", 2.0), ("immediately post", 2.0),
                   ("panic", 1.5), ("trust me", 2.0)]:
        if bad in t:
            raw -= w

    words = len(text.split())
    if words > 140:
        raw -= (words - 140) * 0.03
    elif words < 8:
        raw -= 1.0

    if ideal:
        ideal_m = re.search(r"action\s*:\s*(\w+)", ideal.lower())
        gen_m   = re.search(r"action\s*:\s*(\w+)", t)
        if ideal_m and gen_m and ideal_m.group(1) == gen_m.group(1):
            raw += 2.0

    return float(math.tanh(raw / REWARD_SCALE))


def reward_fn(prompts, completions, ideal=None, **_):
    return [
        compute_reward(c, ideal[i] if ideal and i < len(ideal) else "")
        for i, c in enumerate(completions)
    ]


# ── 4. GRPO trainer ─────────────────────────────────────────────
config = GRPOConfig(
    output_dir                   = "grpo_outputs",
    max_steps                    = MAX_STEPS,
    per_device_train_batch_size  = PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps  = GRAD_ACCUM_STEPS,
    learning_rate                = LEARNING_RATE,
    num_generations              = NUM_GENERATIONS,
    max_completion_length        = MAX_COMPLETION_LEN,
    temperature                  = TEMPERATURE,
    top_p                        = TOP_P,
    beta                         = KL_BETA,
    logging_steps                = 1,
    save_steps                   = 200,
    seed                         = SEED,
)

trainer = GRPOTrainer(
    model         = model,
    tokenizer     = tokenizer,
    reward_funcs  = reward_fn,
    train_dataset = Dataset.from_list(train_prompts),
    args          = config,
)

trainer.train()


# ── 5. plot training curves ─────────────────────────────────────
steps, rewards, kl_vals = [], [], []
for log in trainer.state.log_history:
    if "reward" in log:
        steps.append(log.get("step", len(steps)))
        rewards.append(log["reward"])
        kl_vals.append(log.get("kl", 0))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax1.plot(steps, rewards, marker="o", color="steelblue", label="Reward")
if len(rewards) >= 10:
    rolling = [sum(rewards[max(0, i-10):i+1]) / min(i+1, 10) for i in range(len(rewards))]
    ax1.plot(steps, rolling, color="orange", linewidth=2, label="Rolling avg (10)")
ax1.axhline(0, color="red", linestyle="--", alpha=0.4)
ax1.set_title("GRPO Reward Curve — VeritaRL")
ax1.set_ylabel("Reward")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(steps, kl_vals, color="purple", marker="x")
ax2.set_title("KL Divergence (should grow slowly)")
ax2.set_xlabel("Steps")
ax2.set_ylabel("KL")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("assets/training_curves.png", dpi=120, bbox_inches="tight")
plt.show()


# ── 6. push to HF Hub ───────────────────────────────────────────
print(f"\nPushing merged fp16 checkpoint to {HF_REPO} ...")
model.push_to_hub_merged(
    HF_REPO,
    tokenizer,
    save_method = "merged_16bit",
    private     = False,
)
print(f"Done: https://huggingface.co/{HF_REPO}")
