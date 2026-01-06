# Multi-Label Emotion Recognition under Class Imbalance (GoEmotions + Twitter)

This repository contains a multi-label NLP pipeline for emotion recognition, with a focus on **class imbalance** and **rare emotion detection**. The project uses **GoEmotions (Reddit)** as the primary benchmark and additionally integrates the **dair-ai/emotion (Twitter)** dataset by aligning its labels into the GoEmotions multi-label format.

The core approach is a **TF-IDF + One-vs-Rest Logistic Regression** baseline, followed by **hyperparameter optimization** with `RandomizedSearchCV`. Evaluation emphasizes **label-wise performance** using **Hamming loss** (and optionally F1-based metrics), because subset accuracy can be overly strict for multi-label tasks.

---

## Key Idea

Multi-label emotion datasets are typically **long-tailed**: a small set of emotions dominates the data, while several important emotions appear rarely. Standard models often learn frequent labels well but underperform on rare labels, producing biased predictions.

This project investigates imbalance-aware modeling and evaluation to improve reliability across all emotion labels.

---

## Datasets

- **GoEmotions (Reddit)**: ~58k text samples annotated with **27 emotions + neutral** (28 labels total). Each sample may contain multiple labels.
- **dair-ai/emotion (Twitter)**: ~20k samples with **6 basic emotions**. Originally multi-class (single integer label), reformatted into a multi-label (multi-hot) representation aligned to the GoEmotions label space.

> Note: Dataset downloading and usage must follow their original licenses/terms. This repository does not redistribute the raw datasets.

---

## Methodology

### Baseline
Pipeline:
1. `TfidfVectorizer` for text-to-feature transformation  
2. `OneVsRestClassifier(LogisticRegression)` for multi-label classification (one binary classifier per label)

### Hyperparameter Optimization
A tuned variant uses `RandomizedSearchCV` over the same pipeline to explore:
- **Vectorizer**: `ngram_range`, `min_df`, `max_df`
- **Classifier**: regularization strength `C` (sampled from a log-uniform range), optional `class_weight`

---

## Results (Summary)

Evaluation uses **label-wise accuracy = 1 − Hamming loss**, which measures correctness across all label decisions (rather than requiring a perfect match of all labels per sample).

- **Baseline label-wise accuracy (1 − Hamming loss): 0.960**
- **Tuned label-wise accuracy (1 − Hamming loss): 0.990**

Interestingly, the best tuned configuration performed better **without** `class_weight="balanced"`. A plausible explanation is that weighting can make the model more eager to predict rare labels, which may increase **false positives** and slightly reduce overall label-level correctness in some settings.
