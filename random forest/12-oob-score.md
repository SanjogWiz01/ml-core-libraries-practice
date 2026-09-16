# 12 - Out-of-Bag (OOB) Score

The OOB error is one of Random Forest's most valuable free features.

## What is OOB?

About 36.8% of samples are excluded from each tree's bootstrap sample. These
are the **out-of-bag** samples for that tree — a built-in validation set.

## How OOB Score Works

For each sample `i`:
1. Collect only the trees for which `i` was **out-of-bag**.
2. Predict `i` using only those trees (majority vote / average).
3. Compare against the true label.

Aggregating over all samples gives the **OOB error / score**.

## Why It's Useful

- Estimates test error **without** a separate holdout set.
- Nearly free: no extra training or data split required.
- An honest, nearly unbiased estimator (uses only trees that didn't train on `i`).

## In sklearn

```python
rf = RandomForestClassifier(oob_score=True, random_state=42)
rf.fit(X, y)
print(rf.oob_score_)
```

## OOB vs Cross-Validation

| Method | Cost | Accuracy |
|--------|------|----------|
| K-fold CV | Trains K models | High |
| OOB | Already trained | Slightly optimistic, fast |

OOB is roughly comparable to K-fold but much cheaper — great for quick checks.

## Next Steps

- 13 - Feature Importance
- 27 - Model Evaluation & CV
