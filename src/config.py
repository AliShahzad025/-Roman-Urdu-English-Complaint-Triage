"""Shared settings for the project: paths, label sets, model and training hyper-parameters."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "complaints.csv"
SPLIT_DIR = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "results"
ADAPTER_DIR = Path(os.environ.get("ADAPTER_DIR", ROOT / "outputs" / "complaint-lora"))

SEED = 42
TASKS = ["category", "urgency", "sentiment"]
LANGUAGES = ["en", "roman_urdu", "mixed"]

LABELS = {
    "category": ["damaged_item", "wrong_item", "delivery_issue", "refund_request", "return_exchange",
                 "payment_issue", "cancellation", "general_inquiry", "other"],
    "urgency": ["low", "medium", "high"],
    "sentiment": ["negative", "neutral", "positive"],
}

# Base model. Any small instruct model with a chat template works, e.g. "meta-llama/Llama-3.2-1B-Instruct"
# (gated: needs a Hugging Face token) or "Qwen/Qwen2.5-3B-Instruct" (slower, needs more memory).
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# QLoRA fine-tuning (fits a free Colab/Kaggle T4 GPU).
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EPOCHS = 2
LEARNING_RATE = 2e-4
BATCH_SIZE = 8
GRAD_ACCUM = 2
MAX_SEQ_LEN = 512

# Generation (evaluation and demo): greedy, short JSON answer.
MAX_NEW_TOKENS = 48
EVAL_BATCH_SIZE = 16
