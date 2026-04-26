"""
demo/inference_agent.py

Three modes (auto-detected):
  DEMO mode     -- epsilon-greedy scripted agent, no model needed (instant)
  PARTIAL mode  -- loads a truncated model from available shards
  FULL mode     -- loads the complete GRPO-trained Llama model

To force demo mode:   set RUMOUR_DEMO=1  in your terminal

Optional:
  RUMOUR_QUIET=1       suppress import banner
  RUMOUR_AGENT_LOG=1   print each scripted-agent decision ([DEMO ...])

Requirements for MODEL/PARTIAL mode:
  pip install torch transformers accelerate safetensors
"""

import os
import json
import random as _rng
from pathlib import Path

_THIS_DIR  = Path(__file__).resolve().parent
_PROJECT   = _THIS_DIR.parent
MODEL_PATH = _PROJECT / "models" / "rumor_grpo_model"

_FORCE_DEMO = os.environ.get("RUMOUR_DEMO", "").strip() == "1"
_IMPORT_BANNER = os.environ.get("RUMOUR_QUIET", "").strip() != "1"
_AGENT_LOG = os.environ.get("RUMOUR_AGENT_LOG", "").strip() == "1"


def _agent_log(msg: str) -> None:
    if _AGENT_LOG:
        print(msg)


# -- MODE DETECTION ----------------------------------------------------

def _check_model_ready():
    if _FORCE_DEMO:
        return "demo", "RUMOUR_DEMO=1 set"
    if not MODEL_PATH.exists() or not any(MODEL_PATH.iterdir()):
        return "demo", f"no model at {MODEL_PATH}"

    if not (MODEL_PATH / "config.json").exists():
        return "demo", "config.json missing from model folder"

    has_tokenizer = (MODEL_PATH / "tokenizer.json").exists() or \
                    (MODEL_PATH / "tokenizer_config.json").exists()
    if not has_tokenizer:
        return "demo", "tokenizer files missing"

    index_path = MODEL_PATH / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        needed_shards = set(index["weight_map"].values())
        present = [s for s in needed_shards if (MODEL_PATH / s).exists()]
        missing = [s for s in needed_shards if not (MODEL_PATH / s).exists()]
        if not missing:
            return "full", "all shards present"
        if len(present) >= 2:
            return "partial", f"missing {missing}, will build truncated model"
        return "demo", f"too many shards missing: {missing}"
    elif (MODEL_PATH / "model.safetensors").exists():
        return "full", "single safetensors file"
    else:
        return "demo", "no weight files found"


_MODEL_MODE, _MODEL_STATUS = _check_model_ready()


# -- EPSILON-GREEDY SCRIPTED AGENT -------------------------------------

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
    "layoff":  "warn_team_quietly",
    "cut":     "warn_team_quietly",
    "miss":    "request_budget_freeze",
    "freeze":  "request_budget_freeze",
    "budget":  "request_budget_freeze",
    "q4":      "request_budget_freeze",
    "revenue": "request_budget_freeze",
    "promot":  "escalate_to_leadership",
    "escalat": "escalate_to_leadership",
    "leader":  "escalate_to_leadership",
    "candid":  "escalate_to_leadership",
}


class _ExploringAgent:
    """
    Epsilon-greedy agent that collects evidence and adapts decisions
    based on accumulated signals. Balances exploration vs exploitation.
    """

    def __init__(self, epsilon_start=0.4, epsilon_decay=0.06, epsilon_min=0.05):
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
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
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def _parse_prompt(self, prompt):
        day = 0
        for line in prompt.split("\n"):
            stripped = line.strip()
            if stripped.startswith("DAY") or "Day:" in stripped:
                try:
                    day = int(''.join(filter(str.isdigit, stripped.split("/")[0])))
                except Exception:
                    pass
        return day

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

    def _pick_evidence_decision(self):
        text = self.all_prompt_text
        for keyword, decision in SIGNAL_TO_DECISION.items():
            if keyword in text:
                return decision

        neg = self.episode_signals.count("negative")
        pos = self.episode_signals.count("positive")
        if neg > pos:
            return "warn_team_quietly"
        if pos > neg and neg == 0:
            return "ignore"
        return "wait_for_more_signals"

    def generate(self, prompt):
        day = self._parse_prompt(prompt)
        self._extract_signals(prompt)
        is_exploring = _rng.random() < self.epsilon

        if day <= 2:
            unconsulted = [c for c in CHARACTERS if c not in self.consulted]
            reliable_first = sorted(
                unconsulted,
                key=lambda c: CHARACTERS.index(c)
            )

            if is_exploring and unconsulted:
                target = _rng.choice(unconsulted)
                self.consulted.add(target)
                self.explore_count += 1
                mode = "EXPLORE"
            elif reliable_first:
                target = reliable_first[0]
                self.consulted.add(target)
                self.exploit_count += 1
                mode = "EXPLOIT"
            else:
                self.exploit_count += 1
                _agent_log(f"  [DEMO EXPLOIT] day={day} -> 'wait'")
                return "wait"

            _agent_log(f"  [DEMO {mode}] day={day} -> 'message {target}'")
            return f"message {target}"

        if day == 3 and len(self.consulted) < 2:
            unconsulted = [c for c in CHARACTERS if c not in self.consulted]
            if unconsulted:
                target = unconsulted[0]
                self.consulted.add(target)
                self.exploit_count += 1
                _agent_log(f"  [DEMO EXPLOIT] day={day} -> 'message {target}'")
                return f"message {target}"

        neg = self.episode_signals.count("negative")
        pos = self.episode_signals.count("positive")
        if neg > 0 and pos > 0 and day == 3:
            self.exploit_count += 1
            _agent_log(f"  [DEMO EXPLOIT] day={day} -> 'wait' (conflicting signals)")
            return "wait"

        if is_exploring:
            decision = _rng.choice(DECISIONS)
            self.explore_count += 1
            mode = "EXPLORE"
        else:
            decision = self._pick_evidence_decision()
            self.exploit_count += 1
            mode = "EXPLOIT"

        _agent_log(
            f"  [DEMO {mode}] day={day} -> 'decide {decision}' "
            f"(signals: +{pos}/-{neg}, sources: {len(self.consulted)})"
        )
        return f"decide {decision}"


_agent = _ExploringAgent()


# -- MODEL AGENT -------------------------------------------------------

_model     = None
_tokenizer = None
_device    = None


def _find_available_layers():
    index_path = MODEL_PATH / "model.safetensors.index.json"
    if not index_path.exists():
        return None, None

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    present_shards = {
        s for s in set(weight_map.values()) if (MODEL_PATH / s).exists()
    }

    max_layer = 0
    for key in weight_map:
        if key.startswith("model.layers."):
            layer_num = int(key.split(".")[2])
            max_layer = max(max_layer, layer_num)

    available_layers = []
    for layer_idx in range(max_layer + 1):
        prefix = f"model.layers.{layer_idx}."
        layer_weights = {k: v for k, v in weight_map.items() if k.startswith(prefix)}
        if all(v in present_shards for v in layer_weights.values()):
            available_layers.append(layer_idx)

    has_embed   = weight_map.get("model.embed_tokens.weight", "") in present_shards
    has_norm    = weight_map.get("model.norm.weight", "") in present_shards
    has_lm_head = weight_map.get("lm_head.weight", "") in present_shards

    return available_layers, (has_embed, has_norm, has_lm_head)


def _load_model():
    global _model, _tokenizer, _device
    if _model is not None:
        return True

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
        from safetensors.torch import load_file
    except ImportError as e:
        print(f"[inference_agent] Missing dependency: {e}")
        print("  Run:  pip install torch transformers accelerate safetensors")
        return False

    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"[inference_agent] GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print(f"[inference_agent] No GPU -- loading on CPU (slow but works)...")

    tok_kw = dict(trust_remote_code=True)
    try:
        _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), fix_mistral_regex=True, **tok_kw)
    except TypeError:
        _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), **tok_kw)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    if _MODEL_MODE == "full":
        print(f"[inference_agent] Loading full model...")
        _model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            dtype=torch.float16,
            device_map="auto" if gpu_available else "cpu",
            trust_remote_code=True,
        )
    else:
        avail_layers, (has_embed, has_norm, has_lm_head) = _find_available_layers()
        if not avail_layers or not has_embed or not has_lm_head or not has_norm:
            print("[inference_agent] Not enough weights for a usable truncated model.")
            return False

        contiguous = []
        for layer in sorted(avail_layers):
            if not contiguous or layer == contiguous[-1] + 1:
                contiguous.append(layer)
            else:
                break
        num_layers = len(contiguous)
        print(f"[inference_agent] Building truncated model with {num_layers}/{len(avail_layers)} layers "
              f"(from available shards)...")

        with open(MODEL_PATH / "config.json") as f:
            base_cfg = json.load(f)
        base_cfg["num_hidden_layers"] = num_layers
        base_cfg["use_cache"] = True
        config = LlamaConfig(**{k: v for k, v in base_cfg.items()
                                if k in LlamaConfig().to_dict() or k.startswith("rope")})

        _model = AutoModelForCausalLM.from_config(config)

        index_path = MODEL_PATH / "model.safetensors.index.json"
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        present_shards = {
            s for s in set(weight_map.values()) if (MODEL_PATH / s).exists()
        }

        all_weights = {}
        for shard_name in present_shards:
            print(f"  Loading shard: {shard_name}...")
            shard = load_file(str(MODEL_PATH / shard_name))
            all_weights.update(shard)
            del shard

        state = {}
        state["model.embed_tokens.weight"] = all_weights["model.embed_tokens.weight"]
        state["model.norm.weight"] = all_weights["model.norm.weight"]
        state["lm_head.weight"] = all_weights["lm_head.weight"]

        for new_idx, old_idx in enumerate(contiguous):
            old_prefix = f"model.layers.{old_idx}."
            new_prefix = f"model.layers.{new_idx}."
            for key, tensor in all_weights.items():
                if key.startswith(old_prefix):
                    state[new_prefix + key[len(old_prefix):]] = tensor

        _model.load_state_dict(state, strict=True)
        del all_weights, state
        _model = _model.half()

        if gpu_available:
            _model = _model.cuda()

    _model.config.use_cache = True
    _model.eval()
    _device = next(_model.parameters()).device
    print(f"[inference_agent] Model loaded on {_device}.")
    return True


def _model_generate(prompt, max_new_tokens=64):
    import torch

    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(_device)

    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=_tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# -- PUBLIC API --------------------------------------------------------

def _has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def generate(prompt, max_new_tokens=64):
    """Main entry point. Automatically picks demo, partial, or full model mode."""
    if _MODEL_MODE == "full":
        loaded = _load_model()
        if loaded:
            return _model_generate(prompt, max_new_tokens)
        print("  -> Falling back to DEMO mode.")
    elif _MODEL_MODE == "partial" and _has_gpu():
        loaded = _load_model()
        if loaded:
            return _model_generate(prompt, max_new_tokens)
        print("  -> Falling back to DEMO mode.")

    return _agent.generate(prompt)


def new_episode():
    """Call between episodes to reset agent state and decay epsilon."""
    _agent.new_episode()


def get_agent_stats():
    """Return exploration vs exploitation stats."""
    total = _agent.explore_count + _agent.exploit_count
    return {
        "epsilon": round(_agent.epsilon, 3),
        "episode": _agent.episode_num,
        "explore_count": _agent.explore_count,
        "exploit_count": _agent.exploit_count,
        "explore_rate": round(_agent.explore_count / max(total, 1), 3),
    }


# -- STATUS ON IMPORT --------------------------------------------------
if _IMPORT_BANNER:
    if _MODEL_MODE == "full":
        print(f"[inference_agent] Full model at {MODEL_PATH} (loads on first use).")
    elif _MODEL_MODE == "partial":
        print("[inference_agent] Partial weights: neural model only with GPU; else scripted demo.")
    else:
        print(f"[inference_agent] Scripted demo agent ({_MODEL_STATUS}).")
