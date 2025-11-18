# 📧 Email Spam Detection Using TF-IDF, Logistic Regression & Multinomial Naive Bayes

A complete end‑to‑end **Machine Learning NLP project** for classifying emails as **Spam** or **Ham** using the Kaggle *Spam Mail Dataset*.

This project follows a full DS/ML workflow: data cleaning, EDA, text preprocessing, feature engineering, model building, hyperparameter tuning, model comparison, and ROC‑AUC evaluation.

---

# Project Overview

Email spam detection is a classic NLP classification problem. In this project, I build two models:

* **Logistic Regression (TF‑IDF + bigrams)**
* **Multinomial Naive Bayes (TF‑IDF + bigrams)**

Both are trained and evaluated on a cleaned dataset of ~5000 emails.

The goal is to develop a strong, explainable spam classifier and compare model performance.

---

# Dataset

**Source:** Kaggle — *Spam Mails Dataset(Preprocessed Enron Email Dataset sample), Link:https://www.kaggle.com/datasets/venky73/spam-mails-dataset*

After cleaning:

* **Empty emails removed:** 16
* **Duplicate emails removed:** 178

Final dataset size: **~5,000 emails**

Columns used:

* `label` → spam/ham
* `text` → subject + email body
* `label_num` → 0 = ham, 1 = spam

---

# Data Cleaning

The following steps were applied:

### Removed empty text rows

Some emails contained no content:

```
(df['text'].str.strip() == '').sum() → 16
```

These were removed.

### Removed duplicate emails

```
df.duplicated().sum() → 178
```

Dropped to avoid bias and leakage.

### Recomputed text lengths

* `char_len` → number of characters
* `word_len` → number of words

---

# Exploratory Data Analysis (EDA)

## 1. Length-Based Analysis

* Ham emails are **longer** on average.
* Spam emails tend to be **shorter and templated**, though some spam newsletters are long.

Both **char_len** and **word_len** were used as numeric meta-features.

---

## 2. Token Frequency Analysis

TF‑IDF and CountVectorizer were used to extract linguistic patterns.

### Spam Emails Showed:

* HTML remnants: `font`, `td`, `nbsp`, `width`, `height`
* Links: `http`, `www`, `com`, `href`
* Promotional terms: `free`, `pills`, `price`, `size`
* Scam indicators: `investment advice`, `duty free`, `windows xp`

### Ham Emails Showed:

* Corporate/internal communication: `ect`, `hou`, `enron`
* Business terms: `deal`, `gas`, `meter`, `thanks`
* Thread markers: `original message`, `attached file`

**Clear linguistic separation** → perfect dataset for text classification.

---

# Train/Test Split

```
X_train: (3993,)
X_test:  (999,)
```

Used **Stratified split** to preserve spam/ham balance.

---

# Feature Engineering — TF‑IDF + Numeric Features

A ColumnTransformer was created:

* **TF‑IDF (1–2 grams, max_features=20,000)** on `text`
* **Numeric passthrough:** `char_len`, `word_len`

---

# Models Implemented

Two ML models were trained:

### Logistic Regression

* Solver tuned (`liblinear`)
* C tuned via RandomizedSearchCV
* Works well for sparse high‑dimensional TF‑IDF data

### Multinomial Naive Bayes

* Best suited for TF‑IDF/BoW text
* Tuned `alpha` and `fit_prior`

---

# Hyperparameter Tuning — RandomizedSearchCV

Random search was used due to large vector size.

### Logistic Regression — Best Params

```
{'clf__solver': 'liblinear',
 'clf__penalty': 'l2',
 'clf__C': 233.57}
```

Best CV F1: **0.985**

### Multinomial NB — Best Params

```
{'clf__alpha': 0.01,
 'clf__fit_prior': True}
```

Best CV F1: **0.952**

---

# Model Evaluation

Both evaluations performed on **unseen test set**.

## Logistic Regression (Best Model)

**Accuracy:** 0.99
**F1-Score:** 0.98
**ROC-AUC:** **0.9985**

Confusion Matrix:

```
[[698   9]
 [  3 289]]
```

LR almost perfectly identifies both classes.

---

## Multinomial Naive Bayes

**Accuracy:** 0.92
**F1-Score:** 0.86
**ROC-AUC:** **0.9795**

Confusion Matrix:

```
[[667  40]
 [ 11 281]]
```

MNB performs well but struggles more with spam.

---

# ROC Curves

The ROC curves show:

* Logistic Regression dominates (AUC 0.9985)
* MNB still strong (AUC 0.9795)

Both outperform the baseline.

---

# Final Conclusions

* Logistic Regression with TF‑IDF + bigrams is the **best model**.
* It achieves **99% accuracy** and **near-perfect ROC-AUC**.
* Naive Bayes is simpler but less powerful.
* HTML patterns and promotional keywords were the strongest spam indicators.
* Corporate communication vocabulary clearly separated ham emails.

This project demonstrates a complete, production-quality spam classification workflow.

---

# License

MIT License — you’re free to use and adapt this project with proper credit.
