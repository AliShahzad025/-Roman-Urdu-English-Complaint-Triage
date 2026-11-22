"""Evaluate the small language model on the held-out test set.

  python src/eval_slm.py --name base                  # base model, zero-shot (no fine-tuning)
  python src/eval_slm.py --name finetuned --adapter   # base model + the trained LoRA adapter
  add --limit 50 for a quick check
"""
import argparse
import json
import time

import pandas as pd

from config import ADAPTER_DIR, BASE_MODEL, RESULTS_DIR, SPLIT_DIR, TASKS
from metrics import parse_output, score
from model_utils import generate, load_model, load_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="results are saved as metrics_<name>.json")
    ap.add_argument("--adapter", nargs="?", const=str(ADAPTER_DIR), default=None,
                    help="path of the LoRA adapter (default: outputs/complaint-lora)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_json(SPLIT_DIR / f"{args.split}.jsonl", lines=True)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=0).reset_index(drop=True)
    tok, model = load_tokenizer(), load_model(args.adapter)

    start = time.time()
    raw = generate(model, tok, df["text"].tolist())
    seconds = time.time() - start

    parsed = pd.DataFrame([parse_output(r) for r in raw])
    out = df[["text", "language", *TASKS]].copy()
    out["raw_output"] = raw
    out["json_valid"] = parsed["json_valid"]
    for task in TASKS:
        out[f"pred_{task}"] = parsed[task]

    result = score(out)
    result["model"] = BASE_MODEL + (f" + LoRA ({args.adapter})" if args.adapter else " (zero-shot)")
    result["split"], result["examples"] = args.split, len(out)
    result["seconds_per_example"] = round(seconds / len(out), 3)
    suffix = args.name if not args.limit else f"{args.name}_limit{args.limit}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / f"predictions_{suffix}.csv", index=False, encoding="utf-8")
    (RESULTS_DIR / f"metrics_{suffix}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
