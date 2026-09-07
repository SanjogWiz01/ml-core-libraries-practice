# 14 - Extra Trees vs Random Forest

Extra Trees (Extremely Randomized Trees) is a close cousin you should know.

## What is Extra Trees?

`ExtraTreesClassifier` / `ExtraTreesRegressor` randomize **both** the split
feature **and** the split threshold.

## Key Differences

| Aspect | Random Forest | Extra Trees |
|--------|---------------|-------------|
| Threshold selection | Best threshold from data | Random threshold |
| Node split learning | Optimized | Random (cheap) |
| Bootstrap | Yes | No (uses full dataset) |
| Variance | Lower | Lower still |
| Bias | Moderate | Slightly higher |
| Speed | Slower | Faster |

## Trade-off

- Extra Trees are **faster** to train (no threshold search).
- Can capture **sharper boundaries** since they split at random points.
- Slightly higher bias, but often comparable or better accuracy.

## When to Try It

- Very large datasets where random forests are slow.
- When baseline Random Forest underfits relationships with sharp thresholds.

## sklearn Summary

```python
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_estimators=100, random_state=42)
```

## Next Steps

- 15 - Max Features Tuning
- 16 - Number of Trees
