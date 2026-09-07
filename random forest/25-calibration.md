# 25 - Probability Calibration

Random Forest probabilities are not always well-calibrated. This file explains
when and how to fix it.

## What's Calibration?

A model is calibrated if predictions of ~70% are correct about 70% of the time.
Random Forest averaging of per-tree leaf fractions often gives decent, but not
perfect, calibration — especially with few trees or small leaf sizes.

## Why It Matters

- Threshold tuning (19) relies on trustworthy probabilities.
- Business decisions need honest probabilities (loan default risk, fraud).

## Check Calibration

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

prob_true, prob_pred = calibration_curve(
    y_test, clf.predict_proba(X_test)[:, 1], n_bins=10
)
plt.plot(prob_pred, prob_true, marker='o')  # should hug the diagonal
```

## Fix with CalibratedClassifierCV

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(clf, method='sigmoid' or 'isotonic', cv=5)
calibrated.fit(X_train, y_train)
```

- **Platt / sigmoid**: good when probabilities are miscalibrated monotonically.
- **Isotonic**: non-parametric, better for larger datasets / severe miscalibration.

## Practical Advice (Pareto)

- Calibrate **after** fitting Random Forest, on a separate validation fold.
- Prefer isotonic for big data, sigmoid for small data.
- Re-check calibration on test data after calibration.

## Next Steps

- 21 - Confidence Intervals
- 26 - Classification Metrics