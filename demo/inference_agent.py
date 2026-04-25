"""
demo/inference_agent.py

Two modes:
  DEMO mode  (default, instant) — smart scripted agent, no model needed
  MODEL mode (slow, needs GPU)  — loads your trained LLM from models/

To force demo mode:   set RUMOUR_DEMO=1  in your terminal
To force model mode:  set RUMOUR_DEMO=0  in your terminal

On your laptop → always use DEMO mode (CPU is too slow for 5GB model)
On Colab/GPU   → use MODEL mode
"""

import os
import random
from pathlib import Path

# ── MODE DETECTION ────────────────────────────────────────────
_THIS_DIR  = Path(__file__).resolve().parent
_PROJECT   = _THIS_DIR.parent
MODEL_PATH = _PROJECT / "models" / "rumor_grpo_model"

# Auto-detect: use scripted mode if model folder is missing or RUMOUR_DEMO=1
_FORCE_DEMO  = os.environ.get("RUMOUR_DEMO", "").strip() == "1"
_MODEL_READY = MODEL_PATH.exists() and any(MODEL_PATH.iterdir()) and not _FORCE_DEMO

# ── SCRIPTED AGENT (instant, no GPU needed) ───────────────────
_STEP_COUNTER = [0]  # mutable so nested function can update it

def _scripted_generate(prompt: str) -> str:
    """
    A smart rule-based agent that plays the game correctly.
    Shows meaningful demo output without needing the LLM.
    Day 0 → message quiet_one (most reliable source)
    Day 1 → message leaker    (second most reliable)
    Day 2 → wait              (contradictory signals, gather more)
    Day 3 → message politician (third source)
    Day 4 → decide based on what we've heard
    """
    step = _STEP_COUNTER[0]
    _STEP_COUNTER[0] += 1

    # Extract day from prompt if possible
    day = step  # fallback
    for line in prompt.split("\n"):
        if line.strip().startswith("DAY") or "Day:" in line:
            try:
                day = int(''.join(filter(str.isdigit, line.split("/")[0])))
            except:
                pass

    sequence = [
        "message quiet_one",
        "message leaker",
        "wait",
        "message politician",
        "decide warn_team_quietly",
    ]

    action = sequence[min(day, len(sequence) - 1)]
    print(f"  [DEMO MODE] day={day} → '{action}'")
    return action


# ── MODEL AGENT (slow, needs trained model on disk) ───────────
_model     = None
_tokenizer = None

def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return True

    if not MODEL_PATH.exists():
        print(f"[inference_agent] ⚠️  No model at {MODEL_PATH}")
        return False

    files = list(MODEL_PATH.iterdir())
    if not any(f.name == "config.json" for f in files):
        print(f"[inference_agent] ⚠️  Model folder exists but config.json missing.")
        print(f"  Files found: {[f.name for f in files]}")
        print(f"  → Make sure you copied ALL files from the Colab zip.")
        return False

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[inference_agent] Loading model from {MODEL_PATH} (CPU — this takes ~3 min)...")
    _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    _model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.float16,  # use half precision to reduce RAM usage
        device_map="auto"  # automatically put model on GPU if available
    )
    print("[inference_agent] ✅ Model loaded.")
    return True


def _model_generate(prompt: str, max_new_tokens: int = 32) -> str:
    import torch
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=_tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── PUBLIC API ────────────────────────────────────────────────

def generate(prompt: str, max_new_tokens: int = 32) -> str:
    """
    Main entry point. Automatically picks demo or model mode.
    """
    if _MODEL_READY:
        loaded = _load_model()
        if loaded:
            return _model_generate(prompt, max_new_tokens)
        # fall through to scripted if load failed
        print("  → Falling back to DEMO mode.")

    return _scripted_generate(prompt)


# ── STATUS ON IMPORT ──────────────────────────────────────────
if _MODEL_READY:
    print(f"[inference_agent] Model found at {MODEL_PATH} — will load on first call.")
else:
    reason = "RUMOUR_DEMO=1 set" if _FORCE_DEMO else f"no model at {MODEL_PATH}"
    print(f"[inference_agent] 🎮 DEMO MODE ({reason})")
    print(f"  To use your trained model: copy all Colab zip files to {MODEL_PATH}")