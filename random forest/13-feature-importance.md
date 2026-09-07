# 13 - Feature Importance

Random Forest provides feature importance almost for free — a key reason data
scientists love it.

## What is Feature Importance?

A score indicating how much each feature contributes to predictions. Higher =
more influential.

## Types

### 1. Mean Decrease in Impurity (MDI) — sklearn default
- Sum over all splits of the impurity reduction caused by the feature.
- Weighted by samples reaching the node.
- Prone to bias **toward high-cardinality features**.

```python
importances = rf.feature_importances_
```

### 2. Permutation Importance (more robust)
- Randomly shuffle a feature and measure the drop in score.
- Unbiased toward cardinality; recommended.

```python
from sklearn.inspection import permutation_importance
res = permutation_importance(rf, X_test, y_test, n_repeats=10)
```

## Caveats

- MDI favors continuous / high-cardinality features.
- Highly correlated features split importance between them (under-reporting).
- Importance = predictive usefulness, **not** causal relevance.

## In Practice

- Use for **feature selection** and **interpretability**.
- Prefer permutation importance for reliable rankings.
- Always look at the top-few, not the exact values.

## Next Steps

- 22 - Permutation Importance Deep Dive
- 28 - Feature Selection Workflow
