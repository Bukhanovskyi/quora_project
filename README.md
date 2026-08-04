# Quora Question Pairs — Duplicate Question Detection

Classifying pairs of Quora questions as duplicates or not. This is a binary classification task, evaluated using **Cross-Entropy Loss (log loss)**.

## Task Description

Given pairs of questions (`question1`, `question2`), predict the probability that they are duplicates (`is_duplicate`). The data comes from the original [Kaggle Quora Question Pairs competition](https://www.kaggle.com/c/quora-question-pairs) and is split into train/test at an 80/20 ratio, stratified by `is_duplicate`.

| | Train | Test |
|---|---|---|
| Size | 323 429 | 80 858 |
| Duplicate share | ~36.3% | ~36.3% |

## Repository Structure

```
.
├── p1_preprocessing_and_basic_ML_quora_project.ipynb   # EDA, preprocessing, baseline, TF-IDF, Word2Vec
├── p2_bert_1_frozen.ipynb                               # Frozen BERT embeddings + LogReg/XGBoost
├── p2_bert_2_partial_finetune.ipynb                     # BERT partial fine-tuning (frozen base, unfrozen pooler)
├── p2_bert_3_full_finetune.ipynb                        # BERT full fine-tuning + threshold tuning + ensembling
├── data/                                                 # (mirrored on Google Drive, see below)
│   ├── quora_question_pairs_train.csv                    # raw data (train)
│   ├── quora_question_pairs_test.csv                     # raw data (test)
│   ├── train_processed.csv                               # after preprocessing + engineered features
│   ├── test_processed.csv
│   ├── train_features_full.csv                           # + TF-IDF/Word2Vec similarity features
│   ├── val_features_full.csv
│   ├── test_features_full.csv
│   ├── results_val.csv / results_train.csv                # per-model metric tables (updated by every notebook)
│   └── val_predictions.pkl                                # saved validation probabilities for comparison/ensembling
├── artifacts/                                             # BERT embeddings, feature matrices (Notebook 1 outputs)
└── README.md
```

## Dataset & Artifacts

Due to GitHub's file size limitations, the contents of `data/`, and `artifacts/` are not included in this repository.

All required datasets and generated files are available on Google Drive:

https://drive.google.com/drive/folders/1GsIreY9Y6RKXNlI-sXKERo7otNFrJUt2?usp=sharing

**Note on Notebooks 2 & 3 (BERT fine-tuning):** these were split off from a single file into three separate notebooks specifically to fit within the compute/time limits of a **free Google Colab GPU session**. Each notebook mounts Google Drive directly, reads its inputs from `data/`, and writes its outputs (embeddings, checkpoints, updated `results_*.csv`) back to Drive immediately after computing them.

## Methodology (by notebook)

### `p1_preprocessing_and_basic_ML_quora_project.ipynb`

1. **EDA** — class balance, text length, word-overlap analysis, `qid` repetition across pairs
2. **Preprocessing** — tokenization, stopword removal, stemming (`SnowballStemmer`); engineered
   features based on word overlap (Jaccard, shared words relative to question length)
3. **Train/val split** — Union-Find grouping of questions by `qid` + `GroupShuffleSplit`, to avoid
   data leakage (the same question never appears in both train and val)
4. **Simple baseline** — a threshold on the shared-word ratio (no ML)
5. **TF-IDF + Logistic Regression / XGBoost** — vectorization + similarity features (cosine/euclidean/manhattan)
6. **Word2Vec (`word2vec-google-news-300`) + Logistic Regression / XGBoost** — average and
   TF-IDF-weighted embeddings

### `p2_bert_1_frozen.ipynb`

7. **Frozen BERT embeddings** (`bert-base-uncased`, mean pooling, no fine-tuning) +
   Logistic Regression / XGBoost on top of all previous features

### `p2_bert_2_partial_finetune.ipynb`

8. **BERT partial fine-tuning** — the pretrained base is frozen; only the pooler layer
   (~0.6% of parameters) is trained on top of raw tokenized question pairs, via
   `AutoModelForSequenceClassification` + 🤗 `Trainer`.

### `p2_bert_3_full_finetune.ipynb`

9. **BERT full fine-tuning** — all 109M parameters unfrozen and trained end-to-end
   (`lr=2e-5`, `batch_size=32`, **1 epoch** — see *Key Methodological Findings* below for why).
10. **Decision threshold tuning** — grid search over thresholds on validation to maximize F1,
    since log loss/ROC-AUC are threshold-independent but F1/accuracy are not.
11. **Ensembling** — simple probability averaging of BERT (full fine-tune) with
    XGBoost (Word2Vec), evaluated on validation.
12. **Error analysis** — inspection of false positive/negative examples at the end of every
    notebook, to understand *why* each model fails, not just how often.

## Results (validation)

| Model | Log Loss | F1 | Accuracy | ROC-AUC |
|---|---|---|---|---|
| Naive baseline (constant) | 0.673 | 0.000 | 0.605 | 0.500 |
| Threshold on word_share_jaccard | 0.650 | 0.656 | 0.637 | 0.722 |
| Logistic Regression (TF-IDF) | 0.526 | 0.581 | 0.711 | 0.804 |
| XGBoost (TF-IDF) | 0.521 | 0.523 | 0.700 | 0.798 |
| Logistic Regression (Word2Vec) | 0.532 | 0.581 | 0.700 | 0.790 |
| XGBoost (Word2Vec) | 0.489 | 0.631 | 0.738 | 0.830 |
| Logistic Regression (BERT, frozen) | 0.502 | 0.636 | 0.738 | 0.826 |
| XGBoost (BERT, frozen) | 0.465 | 0.670 | 0.762 | 0.852 |
| BERT (fine-tuned, frozen base) | 0.485 | 0.700 | 0.747 | 0.829 |
| BERT (fine-tuned, full) | 0.408 | 0.783 | 0.838 | 0.917 |
| Ensemble: BERT (full) + XGBoost (Word2Vec) | **0.388** | 0.768 | 0.830 | 0.908 |
| **BERT (fine-tuned, full, tuned threshold)** | 0.408 | **0.806** | **0.841** | **0.917** |

**Best model: BERT, fully fine-tuned, with a tuned decision threshold of 0.26**
(log loss 0.408, F1 0.806, ROC-AUC 0.917 on validation).

The ensemble achieves the lowest log loss of any approach, but a *lower* F1/ROC-AUC than
standalone fine-tuned BERT — see *Key Methodological Findings* for why it wasn't selected
as the final model. On the (leakage-biased) test set, the chosen model scores log loss 0.279,
F1 0.843, ROC-AUC 0.951 — reported here only as a reference value, not as the primary metric
(see the leakage note below).

## Key Methodological Findings

- **Train/test data leakage**: ~36% of questions in `test.csv` also appear in `train.csv`
  (the original split was stratified only by `is_duplicate`, not grouped by `qid`).
  As a result, test metrics are systematically better than val — **val is the primary metric**
  of model quality in this project, and test is reported only as a reference value.
- **Frozen embeddings are a ceiling classical models can't fully overcome.** XGBoost on
  frozen BERT embeddings (log loss 0.465) beat *partial* fine-tuning of BERT itself
  (log loss 0.485) — fine-tuning only a single small layer (the pooler) can't compensate for
  embeddings that were never trained for this specific task. Meaningful gains required
  unfreezing BERT's core representations, i.e. **full fine-tuning**.
- **Full fine-tuning overfits fast — one epoch was already enough.** A run with
  `num_epochs=2` showed training loss keep falling (0.300 → 0.183) while validation loss rose
  (0.420 → 0.467) between epoch 1 and 2 — a clear overfitting signature on a 266k-example
  training set. `load_best_model_at_end=True` protected the reported metrics either way, but
  training a second, discarded epoch wasted ~45 minutes of a free Colab session for no
  benefit, so the final version trains for **1 epoch only**.
- **Threshold tuning is a free win.** The default 0.5 cutoff is not optimal when classes are
  imbalanced (~36% duplicates). Tuning the threshold on validation (best: 0.26) improved F1
  from 0.783 to 0.806 with **zero additional training** — log loss and ROC-AUC are unaffected,
  since they only depend on predicted probabilities, not thresholded labels.
- **Ensembling isn't automatically better.** Averaging fine-tuned BERT's probabilities 50/50
  with the weaker XGBoost (Word2Vec) model improved log loss (0.408 → 0.388) but *hurt* F1
  (0.806 → 0.768) and ROC-AUC (0.917 → 0.908) — a weaker model diluted BERT's more accurate,
  confident predictions. Combining models only pays off when they're of comparable strength,
  or when combined with a weighted average that favors the stronger model.
- **Error analysis reveals two systematic weaknesses in the best model**:
  1. it under-scores duplicate pairs where one question is a much more specific/general version of the other,
  2. it sometimes classifies based on sentence *structure* rather than meaning — e.g. it confidently (87%) predicted "Would you consider Trump a hypocrite?" and "Do you consider
  Trump a pervert?" as duplicates purely because both follow the same "Do/Would you consider X as Y?" template. Some remaining errors also appear to be mislabeled or borderline pairs in the underlying Quora dataset rather than genuine model failures.

## Possible Next Steps

- Try a weighted ensemble average (e.g. `0.7 * bert_proba + 0.3 * xgb_w2v_proba`) instead of a
  flat 50/50 split, to favor the stronger model.
