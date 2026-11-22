"""Real-time testing UI (Gradio).

Tab 1 "Live test": type any message -> TF-IDF, base model and fine-tuned model predictions side by side.
Tab 2 "Test-set challenge": random messages from the held-out test set with the correct labels, a tick or cross
for every prediction, and a running score.

  python app.py            (local)
  python app.py --share    (Colab / Kaggle: prints a public link)

Works before fine-tuning too: the fine-tuned column then shows "not trained yet".
"""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
import pandas as pd

from config import ADAPTER_DIR, BASE_MODEL, LANGUAGES, SPLIT_DIR, TASKS
from metrics import parse_output
from model_utils import generate, load_model, load_tokenizer
from tfidf_baseline import train_classifiers

MODELS = {"tfidf": "TF-IDF + LogReg", "base": "Base SLM (zero-shot)", "finetuned": "Fine-tuned SLM (QLoRA)"}
LANG_NAMES = {"en": "English", "roman_urdu": "Roman Urdu", "mixed": "Mixed"}
EXAMPLES = [
    "parcel toota hua aaya hai, andar glass bhi crack hai 😡",
    "Bhai maine bara size manga tha, chota bhej diya",
    "order kab tak aayega? 5 din ho gaye",
    "mere account se do dafa paise kat gaye, jaldi refund karein warna complaint karunga",
    "jacket ki quality amazing he, thank you so much",
    "Can I exchange the kurta for a larger size?",
    "order cancel kar dein please, ghalti se place ho gaya",
]

print("Training the TF-IDF baseline...")
train_df = pd.read_json(SPLIT_DIR / "train.jsonl", lines=True)
test_df = pd.read_json(SPLIT_DIR / "test.jsonl", lines=True)
tfidf = train_classifiers(train_df)

HAS_ADAPTER = (ADAPTER_DIR / "adapter_config.json").exists()
print(f"Loading {BASE_MODEL}" + (f" + LoRA adapter from {ADAPTER_DIR}" if HAS_ADAPTER else " (no adapter yet)"))
tok = load_tokenizer()
model = load_model(ADAPTER_DIR if HAS_ADAPTER else None)


def run_slm(text: str, use_adapter: bool) -> dict:
    start = time.perf_counter()
    if HAS_ADAPTER and not use_adapter:
        with model.disable_adapter():
            raw = generate(model, tok, [text], batch_size=1)[0]
    else:
        raw = generate(model, tok, [text], batch_size=1)[0]
    return {**parse_output(raw), "raw": raw.strip(), "ms": (time.perf_counter() - start) * 1000}


def predict_all(text: str) -> dict:
    start = time.perf_counter()
    res = {"tfidf": {task: clf.predict([text])[0] for task, clf in tfidf.items()}}
    res["tfidf"]["ms"] = (time.perf_counter() - start) * 1000
    res["base"] = run_slm(text, use_adapter=False)
    if HAS_ADAPTER:
        res["finetuned"] = run_slm(text, use_adapter=True)
    return res


def results_table(res: dict, gold: dict | None = None) -> str:
    rows = ["| Model | Category | Urgency | Sentiment | Time |", "|---|---|---|---|---|"]
    if gold:
        rows.append("| **✔ Correct answer** | " + " | ".join(f"**{gold[t]}**" for t in TASKS) + " | |")
    for key, name in MODELS.items():
        if key not in res:
            rows.append(f"| {name} | *not trained yet* | | | |")
            continue
        r = res[key]
        cells = [r[t] + ("" if gold is None else (" ✅" if r[t] == gold[t] else " ❌")) for t in TASKS]
        rows.append(f"| {name} | " + " | ".join(cells) + f" | {r['ms']:.0f} ms |")
    return "\n".join(rows)


def raw_outputs(res: dict) -> str:
    parts = []
    for key in ("base", "finetuned"):
        if key in res:
            valid = "valid JSON" if res[key]["json_valid"] else "NOT valid JSON"
            parts.append(f"**{MODELS[key]}** ({valid}):\n```\n{res[key]['raw'][:400]}\n```")
    return "\n\n".join(parts)


# ---------- Tab 1: live test ----------
def live(text: str):
    if not text or not text.strip():
        return "Type a message first.", ""
    res = predict_all(text.strip())
    return results_table(res), raw_outputs(res)


# ---------- Tab 2: test-set challenge ----------
def empty_score() -> dict:
    return {key: {"n": 0, "all": 0, **{t: 0 for t in TASKS}} for key in MODELS}


def scoreboard(score: dict) -> str:
    n = score["tfidf"]["n"]
    if n == 0:
        return "*No test messages yet. Click **Next test message**.*"
    rows = [f"**Score after {n} test message{'s' if n > 1 else ''}**", "",
            "| Model | Category | Urgency | Sentiment | All three correct |", "|---|---|---|---|---|"]
    for key, name in MODELS.items():
        s = score[key]
        if s["n"] == 0:
            rows.append(f"| {name} | – | – | – | – |")
            continue
        rows.append(f"| {name} | " + " | ".join(f"{s[t]}/{s['n']}" for t in TASKS) +
                    f" | **{s['all']}/{s['n']}** ({100 * s['all'] / s['n']:.0f}%) |")
    return "\n".join(rows)


def next_message(language: str, score: dict):
    pool = test_df if language == "Any" else test_df[test_df.language == {v: k for k, v in LANG_NAMES.items()}[language]]
    row = pool.iloc[random.randrange(len(pool))]
    gold = {t: row[t] for t in TASKS}
    res = predict_all(row["text"])
    for key, r in res.items():
        s = score[key]
        s["n"] += 1
        for t in TASKS:
            s[t] += int(r[t] == gold[t])
        s["all"] += int(all(r[t] == gold[t] for t in TASKS))
    message = f"### “{row['text']}”\n*Language: {LANG_NAMES.get(row['language'], row['language'])}* · from the held-out test set (never seen in training)"
    return message, results_table(res, gold), raw_outputs(res), scoreboard(score), score


def reset():
    s = empty_score()
    return "", "", "", scoreboard(s), s


status = ("✅ Fine-tuned adapter loaded." if HAS_ADAPTER
          else "⚠️ No fine-tuned adapter found yet (run `src/train_qlora.py`); showing TF-IDF and the base model.")

THEME = gr.themes.Soft()
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6   # Gradio 6 moved `theme` from Blocks() to launch()

with gr.Blocks(title="Roman Urdu Complaint Triage", **({} if GRADIO_6 else {"theme": THEME})) as demo:
    gr.Markdown(f"# Roman Urdu–English Complaint Triage\n"
                f"Base model `{BASE_MODEL}` fine-tuned with QLoRA · labels: **category**, **urgency**, **sentiment**  \n{status}")

    with gr.Tab("Live test"):
        inp = gr.Textbox(label="Customer message (English, Roman Urdu or mixed)", lines=3,
                         placeholder="e.g. parcel toota hua aaya hai, refund chahiye")
        btn = gr.Button("Triage", variant="primary")
        live_table = gr.Markdown()
        with gr.Accordion("Raw model outputs", open=False):
            live_raw = gr.Markdown()
        gr.Examples(EXAMPLES, inp)
        btn.click(live, inp, [live_table, live_raw])
        inp.submit(live, inp, [live_table, live_raw])

    with gr.Tab("Test-set challenge"):
        score_state = gr.State(empty_score())
        with gr.Row():
            lang = gr.Dropdown(["Any", *[LANG_NAMES[l] for l in LANGUAGES]], value="Any", label="Language")
            nxt = gr.Button("Next test message", variant="primary")
            rst = gr.Button("Reset score")
        msg = gr.Markdown()
        test_table = gr.Markdown()
        with gr.Accordion("Raw model outputs", open=False):
            test_raw = gr.Markdown()
        board = gr.Markdown(scoreboard(empty_score()))
        nxt.click(next_message, [lang, score_state], [msg, test_table, test_raw, board, score_state])
        rst.click(reset, None, [msg, test_table, test_raw, board, score_state])

if __name__ == "__main__":
    demo.launch(share="--share" in sys.argv, **({"theme": THEME} if GRADIO_6 else {}))
