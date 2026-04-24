# Track 2 — Multilingual Aspect-Based Sentiment Analysis
### MARBERT Fine-tuning + Hybrid Inference

---

## Overview

This project implements a **hybrid multilingual sentiment analysis pipeline** for aspect-based restaurant review classification. Arabic reviews are routed to a fine-tuned **MARBERT** model, while non-Arabic reviews (English, French, German, etc.) are handled by a pre-trained **XLM-RoBERTa** model. The system outputs per-aspect sentiment labels (`negative`, `neutral`, `positive`) for each review.

---

## Architecture

```
Input Review
     │
     ├─── is_arabic_script? ──► YES ──► MARBERT (fine-tuned)  ──┐
     │                                                            ├──► aspect_sentiments → submission.json
     └──────────────────────── NO  ──► XLM-RoBERTa (zero-shot) ─┘
```

**Aspect Detection** is performed via keyword matching across 5 categories:
- `food`, `service`, `delivery`, `ambiance`, `price`

---

## Repository Structure

```
.
├── 2language-models.ipynb     # Main training + inference notebook
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

---

## Setup

### 1. Environment

Python **3.9+** is required. Install all dependencies:

```bash
pip install -r requirements.txt
```

> **GPU strongly recommended.** The notebook auto-detects CUDA and enables `fp16` training when available.

### 2. Data Paths

The notebook expects data at the following Kaggle input paths. Update the `CONFIG` section at the top of the notebook if running locally:

| Variable        | Default Path                                                                 | Description                        |
|-----------------|------------------------------------------------------------------------------|------------------------------------|
| `TRAIN_PATH`    | `/kaggle/input/datasets/imaisalah/processed2/train_flat_balanced.csv`       | Balanced training CSV              |
| `VAL_PATH`      | `/kaggle/input/datasets/imaisalah/processed2/val_flat.csv`                  | Validation CSV                     |
| `WEIGHTS_PATH`  | `/kaggle/input/datasets/imaisalah/processed2/sentiment_class_weights.json`  | Per-class weight JSON              |
| `TEST_PATH`     | `/kaggle/input/datasets/imaisalah/processed2/unlabeled_clean.csv`           | Unlabeled test CSV                 |

> **Do NOT include the datasets in the ZIP submission.**

Expected CSV columns:
- **Train/Val**: `text_clean`, `sentiment` (`negative` / `neutral` / `positive`)
- **Test**: `review_id`, `text_clean` (or `text`)

### 3. Model Weights

Fine-tuned MARBERT weights are saved automatically during training to:

```
./marbert_output/
```

The best checkpoint (by `macro_f1` on validation) is automatically loaded at the end of training via `load_best_model_at_end=True`.

The XLM-RoBERTa model (`cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`) is loaded directly from Hugging Face Hub — no manual download needed.

---

## How to Run

### On Kaggle

1. Upload the notebook to a Kaggle kernel.
2. Attach the dataset `imaisalah/processed2` as an input.
3. Enable GPU accelerator.
4. Click **Run All**.

### Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Update paths in the CONFIG section of the notebook, then run:
jupyter nbconvert --to notebook --execute 2language-models.ipynb --output executed_output.ipynb
```

---

## Output

The notebook writes predictions to:

```
submission.json
```

**Format:**

```json
[
  {
    "review_id": 42,
    "aspects": ["food", "service"],
    "aspect_sentiments": {
      "food": "positive",
      "service": "neutral"
    }
  }
]
```

---

## Hyperparameters

| Parameter      | Value                                    |
|----------------|------------------------------------------|
| Model (Arabic) | `UBC-NLP/MARBERT`                        |
| Model (Other)  | `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` |
| Max Sequence Length | 128 tokens                          |
| Batch Size     | 16                                       |
| Epochs         | 5 (with early stopping, patience=2)      |
| Learning Rate  | 2e-5                                     |
| Warmup Ratio   | 0.1                                      |
| Weight Decay   | 0.01                                     |
| Loss           | Weighted Cross-Entropy                   |
| Best Metric    | Macro F1                                 |

---

## Notes

- Arabic detection threshold: >20% Arabic Unicode characters in non-space content.
- Class weights are loaded from `sentiment_class_weights.json` to handle label imbalance.
- If no aspect keywords are matched in a review, `food` is used as a fallback aspect.
