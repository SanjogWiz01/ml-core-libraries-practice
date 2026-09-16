# 27 - Cross-Validation for Random Forest

Proper evaluation workflow with K-fold cross-validation.

## Why Not Just One Split?

A single train/test split leaves your estimate at the mercy of randomness.
K-fold CV averages over K splits for a more reliable performance estimate.

## Basic Workflow

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1_macro')
print(scores.mean(), scores.std())
```

## Stratified CV

For classification with imbalance, use **StratifiedKFold** so each fold has
similar class proportions:

```python
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

## Grouped / Time Series Data

- Use `GroupKFold` for grouped samples (same subject).
- For time series, no shuffling: use `TimeSeriesSplit`.

## Leakage Traps

- Fit any imputation / encoding / feature selection **inside** each fold.
- Never fit scalers or selectors on the full dataset before CV.

## Practical Advice (Pareto)

- 5-fold stratified CV is a solid default.
- Report mean ± std of CV scores.
- OOB score (12) can complement CV for a cheap sanity check.

## Next Steps

- 28 - Feature Selection Workflow
- 16 - Number of Trees (tuning by CV)