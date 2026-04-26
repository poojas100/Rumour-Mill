"""Hyperparameters mirroring the Colab GRPO run (see training/train_agent.py).

These are the knobs that actually shipped the checkpoint at
https://huggingface.co/RumorMill/veritarl-tinyllama.
"""

# ── base model ───────────────────────────────────────────────
MODEL_NAME     = "unsloth/tinyllama-chat-bnb-4bit"
HF_REPO        = "RumorMill/veritarl-tinyllama"
MAX_SEQ_LENGTH = 2048

# ── LoRA ─────────────────────────────────────────────────────
LORA_RANK    = 32
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── GRPO ─────────────────────────────────────────────────────
LEARNING_RATE         = 5e-5
MAX_STEPS             = 80
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUM_STEPS      = 8
NUM_GENERATIONS       = 4
MAX_COMPLETION_LEN    = 128
TEMPERATURE           = 0.9
TOP_P                 = 0.95
KL_BETA               = 0.02
SEED                  = 25

# ── reward ───────────────────────────────────────────────────
REWARD_SCALE   = 6.5
VALID_ACTIONS  = ("verify", "gather", "wait", "dismiss", "confirm")
