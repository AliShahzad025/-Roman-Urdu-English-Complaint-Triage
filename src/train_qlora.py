"""QLoRA fine-tuning of the small language model (Hugging Face PEFT).

The model learns to answer the triage prompt with the gold JSON. Loss is computed ONLY on the answer
tokens (the prompt tokens are masked with -100), so the model is not trained to repeat the prompt.
The checkpoint with the lowest validation loss is kept.

  python src/train_qlora.py                 # full run (GPU)
  python src/train_qlora.py --max-steps 3   # smoke test
"""
import argparse
import json
import math

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments, set_seed

from config import (ADAPTER_DIR, BASE_MODEL, BATCH_SIZE, EPOCHS, GRAD_ACCUM, LEARNING_RATE, LORA_ALPHA,
                    LORA_DROPOUT, LORA_R, LORA_TARGETS, MAX_SEQ_LEN, RESULTS_DIR, SEED, SPLIT_DIR)
from model_utils import compute_dtype, load_base_model, load_tokenizer
from prompts import messages_for, target_json


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict], tok):
        self.items = []
        for row in rows:
            prompt = tok.apply_chat_template(messages_for(row["text"]), tokenize=False, add_generation_prompt=True)
            full = tok.apply_chat_template(
                messages_for(row["text"]) + [{"role": "assistant", "content": target_json(row)}], tokenize=False)
            prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            ids = tok(full, add_special_tokens=False)["input_ids"][:MAX_SEQ_LEN]
            assert ids[:len(prompt_ids)] == prompt_ids, "chat template prompt is not a prefix of the full text"
            labels = [-100] * len(prompt_ids) + ids[len(prompt_ids):]
            self.items.append({"input_ids": ids, "labels": labels[:len(ids)]})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(tok):
    def fn(batch):
        width = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            ids.append(b["input_ids"] + [tok.pad_token_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(mask)}
    return fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--train-limit", type=int, default=None)
    args = ap.parse_args()
    set_seed(SEED)

    tok = load_tokenizer()
    tok.padding_side = "right"
    train = pd.read_json(SPLIT_DIR / "train.jsonl", lines=True).sample(frac=1, random_state=SEED)
    val = pd.read_json(SPLIT_DIR / "val.jsonl", lines=True)
    if args.train_limit:
        train, val = train.head(args.train_limit), val.head(max(8, args.train_limit // 4))
    ds_train, ds_val = SFTDataset(train.to_dict("records"), tok), SFTDataset(val.to_dict("records"), tok)
    lengths = [len(x["input_ids"]) for x in ds_train.items]
    print(f"train {len(ds_train)} | val {len(ds_val)} | tokens per example: max {max(lengths)}, "
          f"mean {sum(lengths) / len(lengths):.0f} | answer tokens: {sum(l != -100 for l in ds_train.items[0]['labels'])}")

    model = load_base_model(for_training=True)
    on_gpu = torch.cuda.is_available()
    if on_gpu:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                             target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    dtype = compute_dtype()
    total_steps = args.max_steps if args.max_steps > 0 else \
        math.ceil(len(ds_train) / (BATCH_SIZE * GRAD_ACCUM)) * EPOCHS
    targs = TrainingArguments(
        output_dir=str(ADAPTER_DIR.parent / "checkpoints"), num_train_epochs=EPOCHS, max_steps=args.max_steps,
        per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM, learning_rate=LEARNING_RATE, lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.05 * total_steps)), weight_decay=0.0, logging_steps=10,
        eval_strategy="steps" if args.max_steps > 0 else "epoch",
        save_strategy="steps" if args.max_steps > 0 else "epoch",
        eval_steps=args.max_steps if args.max_steps > 0 else None,
        save_steps=args.max_steps if args.max_steps > 0 else None,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=1, fp16=on_gpu and dtype == torch.float16, bf16=on_gpu and dtype == torch.bfloat16,
        optim="paged_adamw_8bit" if on_gpu else "adamw_torch", report_to="none", seed=SEED,
        remove_unused_columns=False)
    trainer = Trainer(model=model, args=targs, train_dataset=ds_train, eval_dataset=ds_val, data_collator=collate(tok))
    trainer.train()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(ADAPTER_DIR))
    tok.save_pretrained(str(ADAPTER_DIR))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log = {"base_model": BASE_MODEL, "train_examples": len(ds_train), "val_examples": len(ds_val),
           "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "targets": LORA_TARGETS},
           "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "effective_batch": BATCH_SIZE * GRAD_ACCUM,
           "history": trainer.state.log_history}
    (RESULTS_DIR / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Saved LoRA adapter to {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
