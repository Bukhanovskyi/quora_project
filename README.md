# Quora Question Pairs — Duplicate Question Detection

Classifying pairs of Quora questions as duplicates or not. This is a binary classification task,evaluated using **Cross-Entropy Loss (log loss)**.

## Task Description

Given pairs of questions (`question1`, `question2`), predict the probability that they are
duplicates (`is_duplicate`). The data comes from the original [Kaggle Quora Question Pairs competition](https://www.kaggle.com/c/quora-question-pairs)
and is split into train/test at an 80/20 ratio, stratified by `is_duplicate`.

| | Train | Test |
|---|---|---|
| Size | 323 429 | 80 858 |
| Duplicate share | ~36.3% | ~36.3% |

## Repository Structure

```
.
├── p1_preprocessing_and_basic_ML_quora_project.ipynb   # EDA, preprocessing, baseline, TF-IDF, Word2Vec
├── p2_bert.ipynb                                        # Frozen BERT embeddings + LogReg/XGBoost
├── data/
│   ├── quora_question_pairs_train.csv                   # raw data (train)
│   ├── quora_question_pairs_test.csv                    # raw data (test)
│   ├── train_processed.csv                              # after preprocessing + engineered features
│   ├── test_processed.csv
│   ├── train_features_full.csv                          # + TF-IDF/Word2Vec similarity features
│   ├── val_features_full.csv
│   ├── test_features_full.csv
│   ├── results_val.csv / results_train.csv               # per-model metric tables
│   └── val_predictions.pkl                               # saved probabilities for comparison
└── README.md
```
## Dataset

Due to GitHub's file size limitations, the contents of the `data/` folder are not included in this repository.

All required datasets and generated files are available on Google Drive:

https://drive.google.com/drive/folders/1RIzUX4otG9QSWPi8giGe2oN57Vo23rwq?usp=sharing


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

### `p2_bert.ipynb`

7. **Frozen BERT embeddings** (`bert-base-uncased`, mean pooling, no fine-tuning) +
   Logistic Regression / XGBoost on top of all previous features

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
| **XGBoost (BERT, frozen)** | **0.465** | **0.670** | **0.762** | **0.850** |

**Best model: XGBoost on frozen BERT embeddings** (log loss 0.465 on val).
The same model scores 0.390 log loss on test — but this number is **optimistically biased**
(see below).

## Key Methodological Findings

- **Train/test data leakage**: ~36% of questions in `test.csv` also appear in `train.csv`
  (the original split was stratified only by `is_duplicate`, not grouped by `qid`).
  As a result, test metrics are systematically better than val — **val is the primary metric**
  of model quality in this project, and test is reported only as a reference value.
- **The F1@0.5 trap**: models with different probability calibration (LogReg vs XGBoost) cannot
  be fairly compared using F1 at a fixed 0.5 threshold — either tune the optimal threshold
  separately for each model, or rely on log loss/ROC-AUC (which are threshold-independent).
- **Stopwords for Word2Vec**: an isolated test on a subsample showed that removing stopwords
  improved the correlation of a single derived feature (`cosine similarity`) with `is_duplicate`.
  But a full pipeline check showed a **worse** result overall — the standard `nltk` stopword list
  includes question words (`what`, `how`, `why`), which themselves carry signal for this task.
  Stopwords were kept in the final pipeline.
- **XGBoost vs Logistic Regression**: XGBoost is substantially better than LogReg for dense
  embeddings (Word2Vec, BERT), but not for sparse TF-IDF, where results are roughly equal or
  LogReg is slightly better by AUC — a consistent pattern across three different text
  representations.
- **Stemming hurts embeddings**: it raises the out-of-vocabulary rate against the GloVe/Word2Vec
  vocabulary from 1.3% to 19.3% (truncated forms like `"studi"`, `"beauti"` aren't in the
  vocabulary). For TF-IDF, on the other hand, stemming is beneficial.
