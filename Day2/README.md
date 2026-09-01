# Day 2 — Preprocessing Pipelines & Supervised Models

## Overview
Building on Day 1 (problem framing, EDA, 70/10/20 stratified split, and majority-class / rule-based baselines), this notebook builds leak-free `sklearn` pipelines and trains two supervised models — Logistic Regression and a Decision Tree — on the UCI Adult Census Income dataset.

**Primary metric:** ROC AUC (carried over from Day 1's Task 1 justification — handles class imbalance and doesn't lock in one decision threshold).

## Contents
- `adult_census_income_day2.ipynb` — full notebook (preprocessing, training, evaluation, interpretability, write-up)
- `preprocessor.joblib` — saved `ColumnTransformer`
- `log_reg_pipeline.joblib` — saved Logistic Regression pipeline
- `tree_pipeline.joblib` — saved Decision Tree pipeline

## Tasks Completed
1. **Preprocessing plan** — numeric (median impute + scale) and categorical (most-frequent impute + one-hot) pipelines combined via `ColumnTransformer`
2. **Model training** — Logistic Regression and Decision Tree, each fit only on the training split
3. **Evaluation** — accuracy, precision, recall, F1, ROC AUC, PR AUC on the hold-out test set; confusion matrices; ROC/PR curves; comparison against Day 1 baselines
4. **Interpretability** — top logistic regression coefficients; decision tree depth, train/test accuracy, and top splits
5. **Write-up** — model comparison, selection rationale for Day 3, and preprocessing changes planned for tomorrow
```

## Dataset
UCI Adult (Census Income) dataset, loaded via `sklearn.datasets.fetch_openml('adult', version=2)`.

## Next (Day 3)
- Address skew in `capital-gain` / `capital-loss`
- Drop redundant `education` string column
- Simplify `marital-status` grouping
- Try `class_weight='balanced'` given ~24% positive class rate
