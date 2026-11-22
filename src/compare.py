"""Build the comparison report from whatever results exist: table, charts, confusion matrices, error examples.

  python src/compare.py
Writes results/comparison.md, results/f1_by_language.png, results/confusion_category_<model>.png and
results/errors_finetuned.csv / results/fixed_by_finetuning.csv.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from config import LABELS, LANGUAGES, RESULTS_DIR, TASKS

MODELS = [("tfidf", "TF-IDF + LogReg"), ("base", "Base SLM (zero-shot)"), ("finetuned", "Fine-tuned SLM (QLoRA)")]


def load():
    found = {}
    for key, label in MODELS:
        m, p = RESULTS_DIR / f"metrics_{key}.json", RESULTS_DIR / f"predictions_{key}.csv"
        if m.exists() and p.exists():
            found[key] = (label, json.loads(m.read_text(encoding="utf-8")), pd.read_csv(p))
    return found


def table(found) -> str:
    cols = ["macro_f1", "accuracy", *[f"macro_f1_{l}" for l in LANGUAGES], "invalid_rate"]
    lines = []
    for task in TASKS:
        lines += [f"\n### {task}\n", "| Model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for key, (label, m, _) in found.items():
            lines.append(f"| {label} | " + " | ".join(f"{m[task].get(c, float('nan')):.3f}" for c in cols) + " |")
    lines += ["\n### Overall\n", "| Model | all three labels correct | valid JSON | seconds / example |", "|---|---|---|---|"]
    for key, (label, m, _) in found.items():
        lines.append(f"| {label} | {m['all_three_correct']:.3f} | {m.get('json_valid_rate', 1.0):.3f} | "
                     f"{m.get('seconds_per_example', '-')} |")
    return "\n".join(lines)


def charts(found) -> None:
    fig, axes = plt.subplots(1, len(TASKS), figsize=(15, 4), sharey=True)
    width = 0.8 / max(1, len(found))
    for ax, task in zip(axes, TASKS):
        for i, (key, (label, m, _)) in enumerate(found.items()):
            vals = [m[task].get(f"macro_f1_{l}", 0) for l in LANGUAGES]
            ax.bar([x + i * width for x in range(len(LANGUAGES))], vals, width, label=label)
        ax.set_xticks([x + width * (len(found) - 1) / 2 for x in range(len(LANGUAGES))], LANGUAGES)
        ax.set_title(f"{task}: macro-F1 by language"); ax.set_ylim(0, 1)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "f1_by_language.png", dpi=150); plt.close(fig)

    for key, (label, _, p) in found.items():
        fig, ax = plt.subplots(figsize=(9, 8))
        labels = LABELS["category"] + (["INVALID"] if (p["pred_category"] == "INVALID").any() else [])
        ConfusionMatrixDisplay.from_predictions(p["category"], p["pred_category"], labels=labels, ax=ax,
                                                xticks_rotation=60, colorbar=False)
        ax.set_title(f"Category confusion matrix: {label}")
        fig.tight_layout(); fig.savefig(RESULTS_DIR / f"confusion_category_{key}.png", dpi=150); plt.close(fig)


def errors(found) -> None:
    if "finetuned" not in found:
        return
    ft = found["finetuned"][2]
    wrong = ~(ft[[f"pred_{t}" for t in TASKS]].values == ft[TASKS].values).all(axis=1)
    ft[wrong].to_csv(RESULTS_DIR / "errors_finetuned.csv", index=False, encoding="utf-8")
    if "base" in found:
        base = found["base"][2]
        base_wrong = ~(base[[f"pred_{t}" for t in TASKS]].values == base[TASKS].values).all(axis=1)
        fixed = base[base_wrong & ~wrong][["text", "language", *TASKS, *[f"pred_{t}" for t in TASKS]]]
        fixed.to_csv(RESULTS_DIR / "fixed_by_finetuning.csv", index=False, encoding="utf-8")
        print(f"errors: base {base_wrong.sum()}, fine-tuned {wrong.sum()}, fixed by fine-tuning {len(fixed)}, "
              f"broken by fine-tuning {(~base_wrong & wrong).sum()} (of {len(ft)})")


def main() -> None:
    found = load()
    if not found:
        raise SystemExit("No results yet: run tfidf_baseline.py / eval_slm.py first.")
    report = "# Results on the held-out test set\n" + table(found)
    (RESULTS_DIR / "comparison.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    charts(found)
    errors(found)


if __name__ == "__main__":
    main()
