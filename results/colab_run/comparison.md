# Results on the held-out test set

### category

| Model | macro_f1 | accuracy | macro_f1_en | macro_f1_roman_urdu | macro_f1_mixed | invalid_rate |
|---|---|---|---|---|---|---|
| TF-IDF + LogReg | 0.738 | 0.768 | 0.804 | 0.793 | 0.614 | 0.000 |
| Base SLM (zero-shot) | 0.623 | 0.681 | 0.740 | 0.488 | 0.657 | 0.005 |
| Fine-tuned SLM (QLoRA) | 0.912 | 0.917 | 0.991 | 0.814 | 0.960 | 0.000 |

### urgency

| Model | macro_f1 | accuracy | macro_f1_en | macro_f1_roman_urdu | macro_f1_mixed | invalid_rate |
|---|---|---|---|---|---|---|
| TF-IDF + LogReg | 0.765 | 0.767 | 0.846 | 0.831 | 0.640 | 0.000 |
| Base SLM (zero-shot) | 0.408 | 0.535 | 0.464 | 0.350 | 0.362 | 0.005 |
| Fine-tuned SLM (QLoRA) | 0.867 | 0.872 | 0.887 | 0.862 | 0.841 | 0.000 |

### sentiment

| Model | macro_f1 | accuracy | macro_f1_en | macro_f1_roman_urdu | macro_f1_mixed | invalid_rate |
|---|---|---|---|---|---|---|
| TF-IDF + LogReg | 0.836 | 0.867 | 0.880 | 0.930 | 0.696 | 0.000 |
| Base SLM (zero-shot) | 0.690 | 0.769 | 0.716 | 0.655 | 0.644 | 0.005 |
| Fine-tuned SLM (QLoRA) | 0.926 | 0.913 | 0.913 | 0.906 | 0.907 | 0.000 |

### Overall

| Model | all three labels correct | valid JSON | seconds / example |
|---|---|---|---|
| TF-IDF + LogReg | 0.573 | 1.000 | - |
| Base SLM (zero-shot) | 0.339 | 0.995 | 0.807 |
| Fine-tuned SLM (QLoRA) | 0.773 | 1.000 | 0.772 |

errors: base 656, fine-tuned 225, fixed by fine-tuning 478, broken by fine-tuning 47 (of 992)
