# Task 1 Summary: Adult Income Classification

**Web3 Geeks – AI & Data Science Internship**
**Author:** Muhammad Taimoor

---

## 1. Problem Framing

Predicting whether an individual earns more than **$50K/year** using the UCI Adult (Census Income) dataset.

**Business case:** A financial company wants to identify high-income individuals to offer them premium products. Contacting the wrong people wastes budget, but missing too many real targets also costs opportunity — so both **precision** and **recall** matter.

**Dataset:** 48,842 rows
- 11,687 rows (23.9%) earn **>50K**
- 37,155 rows (76.1%) earn **≤50K**

---

## 2. Chosen Metric

**Primary metric:** ROC AUC

With only 23.9% positive class, a model that predicts ≤50K for everyone would already hit 76% accuracy while being useless — so **accuracy is not a usable metric here**.

ROC AUC measures how well the model separates high earners from low earners across all thresholds, without locking in a precision/recall trade-off too early. That trade-off decision gets made later based on actual outreach cost.

**Target for the week:** ROC AUC above **0.88**, with F1 at a 0.5 threshold tracked as a secondary check that the model is practically useful, not just ranking well.

---

## 3. Baseline Results

Hold-out test set, n = 9,769

| Baseline | Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC |
|---|---|---|---|---|---|---|
| Majority Class | 0.761 | 0.000 | 0.000 | 0.000 | 0.500 | 0.239 |
| Rule: `education-num ≥ 13` | 0.753 | 0.484 | 0.497 | 0.491 | 0.719 | 0.444 |

- **Majority class** hits 76% accuracy but scores 0 on precision/recall/F1 (ROC AUC 0.500 — a coin flip), since it never predicts a positive.
- **Education rule** trails on raw accuracy but reaches ROC AUC 0.719 and F1 0.491, catching real high earners at the cost of over-predicting on people with degrees who still earn ≤50K.

---

## 4. Error Analysis & Next Steps

The rule baseline produced **1,237 false positives** and **1,176 false negatives**.

- **False positives** average `education-num` 13.3, age 37.2 — college-educated but earlier-career, often public-sector or part-time roles where a degree hasn't translated to income yet.
- **False negatives** average `education-num` 9.5, age 44.1, and `capital-gain` 2,781 (vs. 196 for false positives) — older people without a Bachelor's who earn well through self-employment or investment income the rule can't see.

False negatives are the **costlier error** for the business, since they represent missed high-value leads.

### Issues to fix before Day 2 modeling

- [ ] Missing values in `workclass`, `occupation`, `native-country` — fill with mode or add an `'Unknown'` category
- [ ] All categorical columns are still strings — one-hot encode low-cardinality, target-encode high-cardinality
- [ ] `capital-gain` / `capital-loss` are heavily skewed — apply `log1p` transform or add a `has_capital_activity` flag
- [ ] `education` and `education-num` are redundant — drop the string column, keep the numeric one
- [ ] `marital-status` has many categories that can be grouped — simplify to `married` / `not-married`
- [ ] Class imbalance (23.9% positive) will affect training — use `class_weight='balanced'` or SMOTE on the training set only

### Plan for the rest of the week

Primary metric remains **ROC AUC**, backed by the baseline numbers above:
- Majority class: 0.500
- Education rule: 0.719
- **Target: 0.88+** from a properly feature-engineered model

Tree-based models (Random Forest, XGBoost) are expected to land around **0.88–0.92**, based on published benchmarks for this dataset. F1 at a 0.5 threshold will be tracked alongside ROC AUC to confirm the model is practically useful, not just ranking well.
