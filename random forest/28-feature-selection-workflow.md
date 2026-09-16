# 28 - Feature Selection Workflow

Leverage Random Forest importance to build leaner, faster, better models.

## Why Select Features

- Cheaper inference, simpler deployment.
- Less noisy features → better generalization.
- Easier interpretation and SHAP analysis.

## Workflow

1. Build a baseline Random Forest.
2. Get importances (MDI 13 or permutation 22).
3. Rank features and inspect the top-k.
4. Retrain on top-k and compare CV scores.
5. Keep the smallest feature set with roughly equal performance.

## Methods

### 1. Threshold on Importance
```python
rf.fit(X_train, y_train)
imp = rf.feature_importances_
keep = X_train.columns[imp > 0.01]
```

### 2. RFE with Random Forest
```python
from sklearn.feature_selection import RFECV
selector = RFECV(rf, step=1, cv=5, scoring='accuracy')
selector.fit(X_train, y_train)
```

### 3. SelectFromModel
```python
from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(rf, threshold='mean')
X_selected = sfm.fit_transform(X_train, y_train)
```

## Pareto Guidance

- The 20/80 applies here too: top ~20% of features usually carry ~80% of the
  signal. Start by keeping the top 10–20 and measuring the impact.
- Prefer permutation importance for the ranking (robust to cardinality).

## Pitfall

- Feature selection must happen **inside** CV folds to stay honest (27).

## Next Steps

- 22 - Permutation Importance
- 13 - Feature Importance