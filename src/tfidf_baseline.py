"""Classic baseline: TF-IDF (character + word n-grams) + Logistic Regression, one classifier per task.

Character n-grams cope with Roman Urdu spelling variations (nahi / nai / nhi). Runs on a CPU in seconds.
Run:  python src/tfidf_baseline.py
"""
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

from config import RESULTS_DIR, SEED, SPLIT_DIR, TASKS
from metrics import score


def train_classifiers(train: pd.DataFrame) -> dict:
    """One TF-IDF + Logistic Regression classifier per task (also used by the demo app)."""
    classifiers = {}
    for task in TASKS:
        clf = make_pipeline(
            make_union(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
                       TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED))
        classifiers[task] = clf.fit(train["text"], train[task])
    return classifiers


def main() -> None:
    train = pd.read_json(SPLIT_DIR / "train.jsonl", lines=True)
    test = pd.read_json(SPLIT_DIR / "test.jsonl", lines=True)
    preds = test[["text", "language", *TASKS]].copy()
    for task, clf in train_classifiers(train).items():
        preds[f"pred_{task}"] = clf.predict(test["text"])
    result = score(preds)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(RESULTS_DIR / "predictions_tfidf.csv", index=False, encoding="utf-8")
    (RESULTS_DIR / "metrics_tfidf.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
