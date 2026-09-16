# 21 - Confidence Intervals & Uncertainty

Random Forest can estimate prediction uncertainty — a valuable, often
under-used feature.

## Source of Uncertainty

Because each tree is trained on a different bootstrap sample and sees different
features, the **spread of tree predictions** reflects model uncertainty.

## Regression: Prediction Interval

For each sample, collect all tree predictions and compute:

- Mean → point prediction.
- Standard deviation → uncertainty.
- Percentiles (e.g., 2.5th / 97.5th) → interval.

```python
import numpy as np

tree_preds = np.array([t.predict(X) for t in rf.estimators_]).T  # (n, T)
mean = tree_preds.mean(axis=1)
low  = np.percentile(tree_preds, 2.5, axis=1)
high = np.percentile(tree_preds, 97.5, axis=1)
```

## Classification: Predicted Probabilities

`predict_proba` yields calibrated-ish class probabilities per sample. The
margin between top-2 classes gives a confidence signal:

- High top-probability + wide margin → confident.
- Near 0.5 → uncertain.

## Caveats

- Tree-variance intervals are **not true Bayesian** intervals; they capture
  model (epistemic) uncertainty, not full data-generating noise.
- Underestimates uncertainty on extrapolation / out-of-distribution data.

## Practical Value (Pareto)

- Use interval width as an **uncertainty score** to flag low-confidence rows for
  human review.
- Combine with OOB (12) and permutation importance (22) for a strong toolkit.

## Next Steps

- 25 - Calibration of Probabilities
- 12 - OOB Score
