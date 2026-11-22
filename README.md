# Fine-Tuning a Small Language Model for Roman Urdu–English Complaint Triage

NLP course project. We fine-tune a small language model (**Qwen2.5-1.5B-Instruct**) with **QLoRA** to label
e-commerce customer complaints, written in English, Roman Urdu or a mix of both, with a **category**, **urgency**
and **sentiment**. The **base model (zero-shot)** and the **fine-tuned model** are compared on the same held-out
test set, overall and per language. A Gradio demo shows both models side by side.

Example input → output:

```
"parcel toota hua aaya hai, andar glass bhi crack hai 😡"
→ {"category": "damaged_item", "urgency": "high", "sentiment": "negative"}
```

## Labels

| Task | Labels |
|---|---|
| category | damaged_item, wrong_item, delivery_issue, refund_request, return_exchange, payment_issue, cancellation, general_inquiry, other |
| urgency | low, medium, high |
| sentiment | negative, neutral, positive |

## Dataset

`data/complaints.csv`: 4,628 labelled customer messages (English 1,524 · Roman Urdu 1,702 · mixed 1,402) in 177
paraphrase groups. It is a copy of the seed dataset from the Resolve AI final-year project; the data is
**synthetic** (template-generated with realistic spelling variation, emojis and greetings).

`src/prepare_data.py` splits it 70 / 10 / 20 into train / validation / test **by paraphrase group**
(`StratifiedGroupKFold`), so no paraphrase of a test complaint is seen in training:

| Split | Messages | Groups |
|---|---|---|
| train | 3,212 | 123 |
| validation | 424 | 18 |
| test | 992 | 36 |

(Split of the reported Colab run. The exact split depends on the scikit-learn version, so a run on another machine can
differ slightly; all models in one run always share the same split. The validation set of this run has no
`general_inquiry` example; it is only used for the validation loss that picks the best epoch.)

## Method

1. **TF-IDF + Logistic Regression** (character 2–5-grams + word 1–2-grams): classic baseline. → `src/tfidf_baseline.py`
2. **Base model, zero-shot**: Qwen2.5-1.5B-Instruct (4-bit) with an instruction prompt that lists the labels and
   their meanings, asked for JSON. → `src/eval_slm.py --name base`
3. **QLoRA fine-tuning**: same model, 4-bit NF4, LoRA r=16 / alpha=32 on all attention and MLP projections,
   2 epochs, lr 2e-4, effective batch 16. Loss only on the answer tokens. Best epoch by validation loss.
   → `src/train_qlora.py`
4. **Fine-tuned model**: same prompt, same test set. → `src/eval_slm.py --name finetuned --adapter`
5. **Comparison**: macro-F1 and accuracy per task, per-language macro-F1 (en / roman_urdu / mixed), exact match on
   all three labels, valid-JSON rate, invalid-label rate, speed; confusion matrices; examples fixed and still wrong
   after fine-tuning. → `src/compare.py`

The base and fine-tuned models use **the same prompt, the same 4-bit quantisation and greedy decoding**, so the
difference between them is the effect of fine-tuning alone.

## Results (held-out test set, 992 messages, Colab T4 run)

| Model | Category macro-F1 | Urgency macro-F1 | Sentiment macro-F1 | All three correct | Valid JSON | Time / message |
|---|---|---|---|---|---|---|
| TF-IDF + LogReg | 0.738 | 0.765 | 0.836 | 57.3% | – | ~10 ms |
| Base SLM (zero-shot) | 0.623 | 0.408 | 0.690 | 33.9% | 99.5% | 0.81 s |
| **Fine-tuned SLM (QLoRA)** | **0.912** | **0.867** | **0.926** | **77.3%** | **100%** | 0.77 s |
| *Fine-tuning gain over base* | *+0.289* | *+0.459* | *+0.236* | *+39.4 pts* | | |

**Category macro-F1 by language**

| Model | English | Roman Urdu | Mixed |
|---|---|---|---|
| TF-IDF + LogReg | 0.804 | 0.793 | 0.614 |
| Base SLM (zero-shot) | 0.740 | **0.488** | 0.657 |
| Fine-tuned SLM (QLoRA) | **0.991** | **0.814** | **0.960** |

Full tables: `results/colab_run/comparison.md` (and the output of step 8 in `notebooks/run_on_colab.ipynb`). Charts from the run:
`results/colab_run/` (F1 by language, category confusion matrices of both models, training curve).

### Analysis

1. **Fine-tuning clearly improves the model.** On the same test set, with the same prompt and quantisation, the
   fine-tuned model beats the base model on every task and every language. The share of messages with all three labels
   correct rises from 33.9% to 77.3%. Of 992 test messages, the base model got 656 wrong (at least one label) and the
   fine-tuned model 225: fine-tuning **fixed 478** and **broke 47**.
2. **The base model struggles most with Roman Urdu and with urgency.** Zero-shot category F1 on Roman Urdu is only
   0.488 (English 0.740), and urgency F1 is 0.408: the base model labels most complaints *high*. Fine-tuning teaches the
   dataset's notion of urgency (urgency F1 0.408 → 0.867) and lifts Roman Urdu category F1 to 0.814.
3. **Fine-tuned vs TF-IDF.** The fine-tuned model is best overall (77.3% vs 57.3% all three correct), with the biggest
   gap on mixed English–Roman Urdu messages (category F1 0.960 vs 0.614). On Roman Urdu alone the two are close
   (category 0.814 vs 0.793), and TF-IDF is slightly better on Roman Urdu sentiment (0.930 vs 0.906): character
   n-grams handle spelling variation well. TF-IDF is also ~80× faster.
4. **Remaining errors** (category confusion matrix, `results/colab_run/confusion_category_finetuned.png`):
   - `wrong_item` → `return_exchange` (27 of 112): "do shoes mangwaye thay, ek hi aaya" (I ordered two, one came) is
     read as a return request.
   - `wrong_item`, `return_exchange`, `payment_issue` → `general_inquiry` (10, 10, 8): complaints phrased as a question
     ("cod ka option show nahi ho raha?").
   - Urgency: the fine-tuned model still tends to say *high* where the gold label is *medium* (calm complaints with a
     greeting or "please").
5. **Over-fitting is visible and handled.** Training loss falls to ~0.0002, while validation loss rises from 0.032
   (epoch 1) to 0.049 (epoch 2) (`results/colab_run/training_curve.png`). The epoch-1 checkpoint (lowest validation
   loss) is the one kept and evaluated. One epoch would be enough, and halves the training time (~2 h on a free T4).

## How to run

**GPU steps (Google Colab, free T4):** upload this folder to Google Drive as `My Drive/NLP Project`
(**leave out the `.venv` folder**), open `notebooks/run_on_colab.ipynb` in Colab, set the runtime to T4 GPU and
run all cells. Measured times on a free T4: base evaluation ~13 min, fine-tuning ~3 h 55 min (2 epochs),
fine-tuned evaluation ~13 min.

**CPU steps (any computer):**

```
python -m venv .venv
.venv\Scripts\pip install pandas scikit-learn matplotlib
.venv\Scripts\python src/prepare_data.py
.venv\Scripts\python src/tfidf_baseline.py
```

**Real-time testing UI:** `python app.py` (add `--share` on Colab for a public link). Two tabs:
- **Live test**: type any message and see TF-IDF, the base model and the fine-tuned model side by side, with response times.
- **Test-set challenge**: random held-out test messages with the correct labels, ✅/❌ for every prediction and a running
  score for each model (filter by language).

Before training finishes, the UI still runs and shows the fine-tuned column as "not trained yet".

To try another base model, set `BASE_MODEL` (e.g. `meta-llama/Llama-3.2-1B-Instruct`) before running the scripts.

## Project structure

```
NLP Project/
├── app.py                     Gradio demo: base vs fine-tuned
├── requirements.txt
├── data/
│   ├── complaints.csv         dataset
│   └── splits/                train / val / test (jsonl)
├── notebooks/run_on_colab.ipynb   runs every step on a Colab GPU
├── src/
│   ├── config.py              paths, labels, model and training settings
│   ├── prompts.py             the instruction prompt (same for both models)
│   ├── prepare_data.py        group-aware split
│   ├── metrics.py             output parsing + scoring (shared by all models)
│   ├── tfidf_baseline.py      classic baseline
│   ├── model_utils.py         4-bit model loading, batched generation
│   ├── eval_slm.py            base / fine-tuned evaluation
│   ├── train_qlora.py         QLoRA fine-tuning
│   └── compare.py             comparison table, charts, error analysis
├── outputs/complaint-lora/    trained LoRA adapter (after training)
└── results/                   metrics, predictions, charts
```

## Limitations

- The paraphrase groups come from templates, so even unseen groups are stylistically close to the training data;
  the very high English scores (category F1 0.991) partly reflect this.
- The dataset is synthetic. Scores on real WhatsApp messages are likely lower; a small real test set would make the
  evaluation stronger.
- Roman Urdu has no standard spelling; unseen spellings and slang can still confuse the model.
- Urgency is partly subjective (the boundary between medium and high), which caps achievable agreement.
