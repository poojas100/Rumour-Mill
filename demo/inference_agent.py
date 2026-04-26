"""
demo/inference_agent.py

Three modes, auto-detected:
  DEMO mode    -- epsilon-greedy scripted agent (instant, no weights needed)
  LOCAL mode   -- loads the merged fp16 checkpoint from models/rumor_grpo_model/
  HUB mode     -- downloads and loads RumorMill/veritarl-tinyllama from HF Hub

Force demo:   RUMOUR_DEMO=1

Environment variables:
  RUMOUR_QUIET=1         suppress import banner
  RUMOUR_AGENT_LOG=1     print scripted-agent decisions
  RUMOUR_LOAD_WARN=1     show tokenizer / triton warnings during load
  RUMOUR_HF_REPO=name    override the HF Hub fallback (default: RumorMill/veritarl-tinyllama)

Requirements for LOCAL/HUB mode:
    pip install torch transformers accelerate safetensors huggingface_hub
"""

import os
import random as _rng
import warnings
from pathlib import Path

_THIS_DIR  = Path(__file__).resolve().parent
_PROJECT   = _THIS_DIR.parent
MODEL_PATH = _PROJECT / "models" / "rumor_grpo_model"
HF_REPO    = os.environ.get("RUMOUR_HF_REPO", "RumorMill/veritarl-tinyllama").strip()

_FORCE_DEMO    = os.environ.get("RUMOUR_DEMO", "").strip() == "1"
_IMPORT_BANNER = os.environ.get("RUMOUR_QUIET", "").strip() != "1"
_AGENT_LOG     = os.environ.get("RUMOUR_AGENT_LOG", "").strip() == "1"
_LOAD_WARN     = os.environ.get("RUMOUR_LOAD_WARN", "").strip() == "1"


def _agent_log(msg: str) -> None:
    if _AGENT_LOG:
        print(msg)


# ── MODE DETECTION ────────────────────────────────────────────

def _local_model_ready() -> bool:
    if not MODEL_PATH.exists() or not any(MODEL_PATH.iterdir()):
        return False
    has_config    = (MODEL_PATH / "config.json").exists()
    has_tokenizer = (MODEL_PATH / "tokenizer.json").exists() or \
                    (MODEL_PATH / "tokenizer_config.json").exists()
    has_weights   = any(p.suffix == ".safetensors" for p in MODEL_PATH.iterdir())
    return has_config and has_tokenizer and has_weights


if _FORCE_DEMO:
    _MODEL_MODE = "demo"
    _MODEL_STATUS = "RUMOUR_DEMO=1 set"
elif _local_model_ready():
    _MODEL_MODE = "local"
    _MODEL_STATUS = f"local weights at {MODEL_PATH}"
else:
    _MODEL_MODE = "hub"
    _MODEL_STATUS = f"no local weights; will try HF Hub ({HF_REPO})"


# ── SCRIPTED AGENT (for DEMO mode & fallback) ────────────────

CHARACTERS = ["quiet_one", "leaker", "politician", "gossip", "spinner"]
DECISIONS  = [
    "warn_team_quietly", "request_budget_freeze",
    "escalate_to_leadership", "wait_for_more_signals", "ignore",
]
SIGNAL_KEYWORDS = {
    "negative": ["layoff", "cut", "miss", "bad", "freeze", "fail", "leaked",
                 "collapsed", "getting cut", "affected", "budget freeze",
                 "numbers missed", "disaster", "ugly"],
    "positive": ["fine", "great", "overblown", "optimistic", "strong",
                 "good", "nothing major", "worry too much"],
}
SIGNAL_TO_DECISION = {
    "layoff": "warn_team_quietly",  "cut": "warn_team_quietly",
    "miss": "request_budget_freeze", "freeze": "request_budget_freeze",
    "budget": "request_budget_freeze", "q4": "request_budget_freeze",
    "revenue": "request_budget_freeze",
    "promot": "escalate_to_leadership", "escalat": "escalate_to_leadership",
    "leader": "escalate_to_leadership", "candid": "escalate_to_leadership",
}


class _ExploringAgent:
    """Epsilon-greedy agent returning raw strings (wait / message X / decide Y)."""

    def __init__(self, eps_start=0.4, eps_decay=0.06, eps_min=0.05):
        self.epsilon = eps_start
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.episode_signals = []
        self.consulted = set()
        self.episode_num = 0
        self.explore_count = 0
        self.exploit_count = 0
        self.all_prompt_text = ""

    def new_episode(self):
        self.episode_signals = []
        self.consulted = set()
        self.all_prompt_text = ""
        self.episode_num += 1
        self.epsilon = max(self.eps_min, self.epsilon - self.eps_decay)

    def _parse_day(self, prompt):
        for line in prompt.split("\n"):
            stripped = line.strip()
            if stripped.startswith("DAY") or "Day " in stripped:
                try:
                    return int("".join(filter(str.isdigit, stripped.split("/")[0])))
                except Exception:
                    pass
        return 0

    def _extract_signals(self, prompt):
        low = prompt.lower()
        self.all_prompt_text += " " + low
        for kw in SIGNAL_KEYWORDS["negative"]:
            if kw in low:
                self.episode_signals.append("negative")
                return
        for kw in SIGNAL_KEYWORDS["positive"]:
            if kw in low:
                self.episode_signals.append("positive")
                return

    def _evidence_decision(self):
        for kw, dec in SIGNAL_TO_DECISION.items():
            if kw in self.all_prompt_text:
                return dec
        neg = self.episode_signals.count("negative")
        pos = self.episode_signals.count("positive")
        if neg > pos:
            return "warn_team_quietly"
        if pos > neg and neg == 0:
            return "ignore"
        return "wait_for_more_signals"

    def generate(self, prompt):
        day = self._parse_day(prompt)
        self._extract_signals(prompt)
        is_exploring = _rng.random() < self.epsilon

        if day <= 2:
            unconsulted = [c for c in CHARACTERS if c not in self.consulted]
            if is_exploring and unconsulted:
                target = _rng.choice(unconsulted)
                self.consulted.add(target)
                self.explore_count += 1
                _agent_log(f"  [DEMO EXPLORE] day={day} -> message {target}")
                return f"message {target}"
            if unconsulted:
                target = unconsulted[0]
                self.consulted.add(target)
                self.exploit_count += 1
                _agent_log(f"  [DEMO EXPLOIT] day={day} -> message {target}")
                return f"message {target}"
            self.exploit_count += 1
            return "wait"

        neg = self.episode_signals.count("negative")
        pos = self.episode_signals.count("positive")
        if day == 3 and neg > 0 and pos > 0:
            self.exploit_count += 1
            _agent_log(f"  [DEMO EXPLOIT] day={day} -> wait (conflicting)")
            return "wait"

        if is_exploring:
            decision = _rng.choice(DECISIONS)
            self.explore_count += 1
            _agent_log(f"  [DEMO EXPLORE] day={day} -> decide {decision}")
        else:
            decision = self._evidence_decision()
            self.exploit_count += 1
            _agent_log(f"  [DEMO EXPLOIT] day={day} -> decide {decision}")
        return f"decide {decision}"


_agent = _ExploringAgent()


# ── MODEL LOADING (LOCAL or HUB) ─────────────────────────────

_model     = None
_tokenizer = None
_device    = None


def _load_model() -> bool:
    global _model, _tokenizer, _device
    if _model is not None:
        return True

    try:
        import torch
        from contextlib import nullcontext
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"[inference_agent] Missing dependency: {e}")
        print("  pip install torch transformers accelerate safetensors huggingface_hub")
        return False

    source = str(MODEL_PATH) if _MODEL_MODE == "local" else HF_REPO
    print(f"[inference_agent] Loading TinyLlama checkpoint from: {source}")

    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[inference_agent] GPU: {gpu} ({vram:.1f} GB)")
    else:
        print("[inference_agent] No GPU — loading TinyLlama (~2 GB) on CPU.")

    wctx = nullcontext() if _LOAD_WARN else warnings.catch_warnings()
    with wctx:
        if not _LOAD_WARN:
            warnings.simplefilter("ignore")

        try:
            _tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
            _model = AutoModelForCausalLM.from_pretrained(
                source,
                dtype=torch.float16 if gpu_available else torch.float32,
                device_map="auto" if gpu_available else "cpu",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[inference_agent] Model load failed: {e!r}")
            print("  Falling back to scripted demo agent.")
            return False

    _model.eval()
    _device = next(_model.parameters()).device
    print(f"[inference_agent] Loaded on {_device}.")
    return True


# ── MODEL GENERATION ─────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a Rumour Intelligence Agent responsible for protecting organizational reputation.\n\n"
    "When presented with a rumour, you must respond in this EXACT format:\n"
    "ACTION: [Verify / Gather / Wait / Dismiss / Confirm]\n"
    "STEP: [One concrete next step to take]\n"
    "RATIONALE: [Why this action protects reputation]\n\n"
    "Never act immediately on unverified claims. Always prioritize verification."
)


def build_veritarl_prompt(day: int, social: float, signals: list[str]) -> str:
    """Zephyr-style prompt matching what the trained model saw during GRPO."""
    if _tokenizer is None:
        eos = "</s>"
    else:
        eos = _tokenizer.eos_token or "</s>"

    sig_text = "\n".join(f"- {s}" for s in signals) or "- (no new signals today)"
    user = (
        f"Day {day} | Reputation score: {int(social)}\n\n"
        f"Incoming signals:\n{sig_text}\n\n"
        f"What do you do?"
    )
    return (
        f"<|system|>\n{SYSTEM_PROMPT}{eos}\n"
        f"<|user|>\n{user}{eos}\n"
        f"<|assistant|>\n"
    )


def _model_generate(prompt: str, max_new_tokens: int = 128) -> str:
    import torch

    inputs = _tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=2048,
    ).to(_device)

    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=_tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(gen, skip_special_tokens=True).strip()


# ── PUBLIC API ───────────────────────────────────────────────

def generate(prompt: str, max_new_tokens: int = 128) -> str:
    """
    Returns either:
      * structured `ACTION: ... / STEP: ... / RATIONALE: ...` (when the trained
        model is loaded), or
      * scripted `wait` / `message X` / `decide Y` (demo fallback).

    `demo/sample_episodes.py` normalises both forms into a `RumorAction`.
    """
    if _MODEL_MODE in ("local", "hub"):
        if _load_model():
            return _model_generate(prompt, max_new_tokens)
        print("  -> Falling back to DEMO mode.")
    return _agent.generate(prompt)


def new_episode() -> None:
    _agent.new_episode()


def get_agent_stats() -> dict:
    total = _agent.explore_count + _agent.exploit_count
    return {
        "epsilon":       round(_agent.epsilon, 3),
        "episode":       _agent.episode_num,
        "explore_count": _agent.explore_count,
        "exploit_count": _agent.exploit_count,
        "explore_rate":  round(_agent.explore_count / max(total, 1), 3),
    }


# ── IMPORT BANNER ────────────────────────────────────────────
if _IMPORT_BANNER:
    if _MODEL_MODE == "local":
        print(f"[inference_agent] Local TinyLlama at {MODEL_PATH} (loads on first use).")
    elif _MODEL_MODE == "hub":
        print(f"[inference_agent] Will pull {HF_REPO} from HF Hub on first use.")
    else:
        print(f"[inference_agent] Scripted demo agent ({_MODEL_STATUS}).")
