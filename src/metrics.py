"""Parsing model output and scoring predictions (shared by every model, so all are scored the same way)."""
import json
import re

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from config import LABELS, LANGUAGES, TASKS

INVALID = "INVALID"
_JSON = re.compile(r"\{.*?\}", re.DOTALL)


def _norm(value) -> str:
    return re.sub(r"[\s\-]+", "_", str(value).strip().lower())


def parse_output(raw: str) -> dict:
    """Turn generated text into {category, urgency, sentiment, json_valid}.

    A label that is missing or not in the allowed set becomes INVALID (scored as wrong)."""
    parsed, valid = {}, False
    match = _JSON.search(raw or "")
    if match:
        try:
            parsed, valid = json.loads(match.group(0)), True
        except json.JSONDecodeError:
            parsed = {}
    if not isinstance(parsed, dict):
        parsed, valid = {}, False
    out = {"json_valid": valid}
    for task in TASKS:
        value = _norm(parsed.get(task, ""))
        if value not in LABELS[task]:
            # fall back to a "key": "value" pattern in broken JSON
            m = re.search(rf'"?{task}"?\s*:\s*"?([a-zA-Z_ \-]+)"?', raw or "")
            value = _norm(m.group(1)) if m else ""
        out[task] = value if value in LABELS[task] else INVALID
    return out


def macro_f1(y, p) -> float:
    """Macro-F1 over the classes that occur in the gold labels of this (sub)set.

    Classes absent from the subset are left out (otherwise each would count as F1 = 0). Wrong or INVALID
    predictions still count as errors through recall and precision of the gold classes."""
    present = [label for label in dict.fromkeys(y)]
    return f1_score(y, p, labels=present, average="macro", zero_division=0)


def score(df: pd.DataFrame) -> dict:
    """df has columns language, <task> (gold) and pred_<task>. Returns overall and per-language scores."""
    out = {}
    for task in TASKS:
        y, p = df[task].tolist(), df[f"pred_{task}"].tolist()
        res = {"macro_f1": macro_f1(y, p), "accuracy": accuracy_score(y, p),
               "invalid_rate": sum(v == INVALID for v in p) / len(p)}
        for lang in LANGUAGES:
            part = df[df.language == lang]
            if len(part):
                res[f"macro_f1_{lang}"] = macro_f1(part[task].tolist(), part[f"pred_{task}"].tolist())
                res[f"accuracy_{lang}"] = accuracy_score(part[task], part[f"pred_{task}"])
        out[task] = {k: round(float(v), 4) for k, v in res.items()}
    all_three = (df[[f"pred_{t}" for t in TASKS]].values == df[TASKS].values).all(axis=1)
    out["all_three_correct"] = round(float(all_three.mean()), 4)
    if "json_valid" in df:
        out["json_valid_rate"] = round(float(df["json_valid"].mean()), 4)
    return out
