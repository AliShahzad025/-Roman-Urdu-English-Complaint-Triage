"""Split the dataset into train / validation / test (70 / 10 / 20) by paraphrase group.

Each complaint in the dataset has ~26 paraphrases that share a group_id. Splitting by group keeps all
paraphrases of a complaint in ONE split, so the test set measures generalisation to unseen complaints,
not memorisation of near-duplicates. StratifiedGroupKFold keeps every category in every split.

Run:  python src/prepare_data.py
"""
import json

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import DATA_CSV, LABELS, RESULTS_DIR, SEED, SPLIT_DIR, TASKS


def group_split(frame: pd.DataFrame, n_splits: int, seed: int):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    keep, held = next(sgkf.split(frame, frame["category"], groups=frame["group_id"]))
    return frame.iloc[keep].reset_index(drop=True), frame.iloc[held].reset_index(drop=True)


def main() -> None:
    df = pd.read_csv(DATA_CSV).dropna(subset=["text", *TASKS, "language", "group_id"])
    df["text"] = df["text"].str.strip()
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    for task in TASKS:
        unknown = set(df[task]) - set(LABELS[task])
        assert not unknown, f"{task}: labels not in config.LABELS: {unknown}"

    trainval, test = group_split(df, n_splits=5, seed=SEED)      # 20% test
    train, val = group_split(trainval, n_splits=8, seed=SEED)    # 12.5% of the rest = 10% val

    assert not (set(train.group_id) & set(test.group_id)) and not (set(val.group_id) & set(test.group_id))
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_json(SPLIT_DIR / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
        stats[name] = {"rows": len(part), "groups": part.group_id.nunique(),
                       **{c: part[c].value_counts().to_dict() for c in ["language", *TASKS]}}
        missing = {t: sorted(set(LABELS[t]) - set(part[t])) for t in TASKS}
        missing = {t: m for t, m in missing.items() if m}
        print(f"{name:5s} rows={len(part):5d} groups={part.group_id.nunique():3d} "
              f"languages={part.language.value_counts().to_dict()}" + (f"  MISSING {missing}" if missing else ""))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote {SPLIT_DIR}/{{train,val,test}}.jsonl and results/dataset_stats.json")


if __name__ == "__main__":
    main()
