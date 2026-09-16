# 22 - Permutation Importance Deep Dive

Permutation importance is the most reliable feature-scoring method for Random
Forest.

## Motivation

MDI (sklearn's `feature_importances_`) is **biased** toward high-cardinality
features and can mis-rank features when they're correlated. Permutation
importance is model-agnostic and unbiased.

## How It Works

For each feature:
1. Measure baseline score (say, accuracy) on OOB or held-out data.
2. Randomly **shuffle** that feature's column (breaks its relationship to y).
3. Re-measure score; the **drop in score** = importance of that feature.

A big drop → the model relied on the feature. A small drop → irrelevant.

## In sklearn

```python
from sklearn.inspection import permutation_importance

res = permutation_importance(
    rf, X_test, y_test, n_repeats=10, random_state=42
)
print(res.importances_mean)
print(res.importances_std)   # variance across repeats
```

## Interpretation Best Practices

- Shuffle within the same dataset (not a fresh sample) → robust to new data.
- Use the **standard deviation** to check reliability of the ranking.
- Correlated features: shuffling one barely hurts because the other compensates
  → correlated groups are under-reported. Shuffle **groups** to fix.

## When to Prefer Permutation

- High-cardinality categoricals present.
- Need robust ranking for feature selection.
- Comparing importance across models.

## Next Steps

- 13 - Feature Importance Basics
- 28 - Feature Selection Workflow