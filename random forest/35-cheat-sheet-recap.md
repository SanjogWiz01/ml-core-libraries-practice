# 35 - Cheat Sheet & Quick Recap

A one-page summary of the entire series, engineered with the Pareto principle.

## What is Random Forest?

Ensemble of decorrelated decision trees (bootstrap + random feature subsets),
combined by voting (classification) or averaging (regression).

## The 20% You Must Know

| Concept | File |
|---------|------|
| Bootstrap sampling | 03 |
| Random feature subspace | 04 |
| Bagging / variance reduction | 05 |
| Two randomnesses = decorrelation | 11 |
| Importance (MDI + permutation) | 13, 22 |
| The 3 knobs to tune | 15, 16, 17 |
| OOB score (free validation) | 12 |
| Class imbalance fix | 19 |
| Honest metrics | 26, 29 |

## Instant sklearn Recipe

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=300,
    max_features='sqrt',
    min_samples_leaf=3,
    class_weight='balanced',
    oob_score=True,
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)
print(rf.oob_score_)
```

## Tuning Order (Pareto 20/80)

1. `max_features` (biggest lever)
2. `min_samples_leaf` / `min_samples_split`
3. `n_estimators` (until plateau)
4. `max_depth` only if overfitting

## Quick Answers

- **Scaling?** No (20).
- **Missing data?** Impute or flag (18).
- **Categoricals?** Encode (20).
- **Imbalance?** `class_weight='balanced'` + PR-AUC (19, 26).
- **Explain?** SHAP + PDP (23, 24).
- **Better than boosting?** Sometimes; boosting on big data (30).

## Full File Map (this series)

1–05: Core theory  06–09: Hands-on basics  10–17: Hyperparameters
18–20: Data prep  21–25: Understanding/calibration  26–29: Metrics + CV
30–32: Comparisons + applications  33–34: Workflow + pitfalls  35: You are here