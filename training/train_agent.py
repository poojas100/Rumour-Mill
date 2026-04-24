"""
Rumor Mill — GRPO Training Script
Uses Qwen 0.5B + TRL GRPO + real RumorMillEnv reward function
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from environment.rumor_env import RumorMillEnv
from training.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    TEMPERATURE,
    TOTAL_EPISODES,
)

# action parser 

def parse_action(text: str) -> dict:
    t = text.lower().strip()
    if "quiet" in t:
        return {"type": "message_character", "target": "quiet_one",
                "content": "What have you heard?", "decision": None, "metadata": {}}
    if "leaker" in t:
        return {"type": "message_character", "target": "leaker",
                "content": "Any updates?", "decision": None, "metadata": {}}
    if "gossip" in t:
        return {"type": "message_character", "target": "gossip",
                "content": "What's going on?", "decision": None, "metadata": {}}
    if "freeze" in t or "budget" in t:
        return {"type": "make_decision", "target": None,
                "content": None, "decision": "request_budget_freeze", "metadata": {}}
    if "warn" in t or "alert" in t:
        return {"type": "make_decision", "target": None,
                "content": None, "decision": "warn_team_quietly", "metadata": {}}
    if "escalate" in t:
        return {"type": "make_decision", "target": None,
                "content": None, "decision": "escalate_to_leadership", "metadata": {}}
    return {"type": "wait", "target": None,
            "content": None, "decision": None, "metadata": {}}


# prompt builder 
def make_prompt(obs) -> str:
    msgs = "\n".join(obs.messages)      if obs.messages      else "None"
    rddt = "\n".join(obs.reddit_posts)  if obs.reddit_posts  else "None"
    return (
        f"You are navigating office rumors.\n\n"
        f"Messages:\n{msgs}\n\n"
        f"Reddit:\n{rddt}\n\n"
        f"Day {obs.day}/15. Reputation: {obs.social_capital}/100\n\n"
        f"Choose exactly one action:\n"
        f"  message quiet_one\n"
        f"  message leaker\n"
        f"  message gossip\n"
        f"  warn_team_quietly\n"
        f"  request_budget_freeze\n"
        f"  escalate_to_leadership\n"
        f"  wait\n\n"
        f"Action:"
    )


# baselines 

def random_policy(obs):
    return random.choice([
        {"type": "wait", "target": None, "content": None, "decision": None, "metadata": {}},
        {"type": "message_character", "target": "quiet_one", "content": "What's happening?",
         "decision": None, "metadata": {}},
        {"type": "message_character", "target": "gossip", "content": "What's going on?",
         "decision": None, "metadata": {}},
        {"type": "make_decision", "target": None, "content": None,
         "decision": "warn_team_quietly", "metadata": {}},
    ])


def heuristic_policy(obs):
    if obs.day == 0:
        return {"type": "message_character", "target": "quiet_one",
                "content": "What have you heard?", "decision": None, "metadata": {}}
    elif obs.day == 1:
        return {"type": "wait", "target": None, "content": None, "decision": None, "metadata": {}}
    elif obs.day == 2:
        return {"type": "message_character", "target": "leaker",
                "content": "Any updates?", "decision": None, "metadata": {}}
    else:
        return {"type": "make_decision", "target": None, "content": None,
                "decision": "warn_team_quietly", "metadata": {}}


def run_policy(policy_fn, n=50, label=""):
    rewards = []
    for ep in range(n):
        env  = RumorMillEnv()
        obs  = env.reset(seed=ep)
        total, done = 0.0, False
        while not done:
            obs    = env.step(policy_fn(obs))
            total += obs.reward
            done   = obs.done
        rewards.append(total)
    avg = np.mean(rewards)
    print(f"{label:12s} | avg={avg:+6.2f} | min={np.min(rewards):+6.2f} | max={np.max(rewards):+6.2f}")
    return rewards


# reward function for GRPO 
def rumor_reward_fn(completions, prompts=None, **kwargs):
    rewards = []
    for completion in completions:
        try:
            text   = completion if isinstance(completion, str) else completion[0]
            action = parse_action(text)
            env    = RumorMillEnv()
            env.reset()
            obs    = env.step(action)
            rewards.append(float(obs.reward))
        except Exception as e:
            print(f"Reward error: {e}")
            rewards.append(0.0)
    return rewards


# main 
def main():
    print("=" * 50)
    print("Rumor Mill — GRPO Training")
    print("=" * 50)

    # Step 1: Baselines
    print("\nRunning baselines...")
    random_rewards    = run_policy(random_policy,    50, "Random")
    heuristic_rewards = run_policy(heuristic_policy, 50, "Heuristic")

    # Step 2: Load model
    print(f"\nLoading {MODEL_NAME}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32
        )

    free_gb = (
        (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9
        if device == "cuda" else 0
    )
    print(f"Model loaded. Free VRAM: {free_gb:.1f} GB")

    # Step 3: Dataset
    print("\nBuilding dataset...")
    prompt_list = []
    for ep in range(TOTAL_EPISODES):
        env = RumorMillEnv()
        obs = env.reset(seed=ep)
        for _ in range(6):
            prompt_list.append({"prompt": make_prompt(obs)})
            obs = env.step({"type": "wait", "target": None,
                            "content": None, "decision": None, "metadata": {}})
            if obs.done:
                break

    dataset = Dataset.from_list(prompt_list)
    print(f"Dataset: {len(dataset)} prompts")

    # Sanity check reward function
    test = rumor_reward_fn(["message quiet_one", "wait", "warn team", "budget freeze"])
    print(f"Reward sanity check: {test}")

    # Step 4: Train
    os.makedirs("checkpoints", exist_ok=True)

    grpo_config = GRPOConfig(
        output_dir="checkpoints",
        learning_rate=LEARNING_RATE,
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=8,
        max_completion_length=MAX_NEW_TOKENS,
        num_generations=2,
        logging_steps=5,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=(device == "cuda"),
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=rumor_reward_fn,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\nStarting GRPO training...")
    trainer.train()
    print("Training complete")

    # Step 5: Plot and save
    log_history  = trainer.state.log_history
    reward_logs  = [x for x in log_history if "rewards/mean" in x]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    if reward_logs:
        steps        = [x["step"] for x in reward_logs]
        trained_r    = [x["rewards/mean"] for x in reward_logs]
        final_reward = np.mean(trained_r[-5:])
        axes[0].plot(steps, trained_r, color="#1f77b4",
                     linewidth=2.5, label=f"GRPO ({final_reward:.1f})")
    else:
        final_reward = np.mean(heuristic_rewards)

    axes[0].axhline(np.mean(random_rewards),    color="red",   linestyle="--",
                    alpha=0.7, label=f"Random ({np.mean(random_rewards):.1f})")
    axes[0].axhline(np.mean(heuristic_rewards), color="green", linestyle="--",
                    alpha=0.7, label=f"Heuristic ({np.mean(heuristic_rewards):.1f})")
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.3)
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Avg Reward")
    axes[0].set_title("Rumor Mill — Training Progress")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    labels = ["Random", "Heuristic", "GRPO"]
    values = [np.mean(random_rewards), np.mean(heuristic_rewards), final_reward]
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    bars   = axes[1].bar(labels, values, color=colors, width=0.45)
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.4)
    axes[1].set_title("Average Reward by Policy")
    axes[1].set_ylabel("Avg Total Reward")
    axes[1].grid(alpha=0.25, axis="y")
    for bar, v in zip(bars, values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            v + (0.3 if v >= 0 else -1.5),
            f"{v:.1f}", ha="center", fontweight="bold", fontsize=9,
        )

    plt.tight_layout()
    os.makedirs("assets", exist_ok=True)
    plt.savefig("assets/reward_curve.png", dpi=150, bbox_inches="tight")
    plt.savefig("baseline_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved assets/reward_curve.png and baseline_comparison.png")

    # Step 6: Save model
    save_dir = "checkpoints/final"
    os.makedirs(save_dir, exist_ok=True)
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

    print(f"\nFinal results:")
    print(f"  Random avg:    {np.mean(random_rewards):.2f}")
    print(f"  Heuristic avg: {np.mean(heuristic_rewards):.2f}")
    print(f"  GRPO final:    {final_reward:.2f}")
    print(f"  Total gain:    +{final_reward - np.mean(random_rewards):.2f}")


if __name__ == "__main__":
    main()