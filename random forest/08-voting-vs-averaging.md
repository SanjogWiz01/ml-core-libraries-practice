# 08 - Voting vs Averaging Aggregation

This file details how Random Forest combines tree outputs.

## Classification: Voting

### Hard Voting
Each tree predicts a class label; the mode (most frequent) is chosen.

### Soft Voting
Each tree outputs class probabilities; these are averaged to get overall class
probabilities; the class with the highest average probability is chosen.

- Soft voting often performs better on balanced, confident models.
- Hard voting is simpler and robust to poorly calibrated probabilities.

## Regression: Averaging

Regressors average the numeric predictions:

```
y_final = mean(y_1, y_2, ..., y_T)
```

Because trees are noisy but unbiased, the mean reduces variance.

## Tie-breaking

- sklearn picks the class with the smallest index on ties.
- For multiclass, `predict` returns argmax of mean probabilities.

## numpy Behind the Scenes

```python
import numpy as np
votes = np.array([t.predict(X) for t in trees])  # (T, n)
pred = np.apply_along_axis(lambda v: np.bincount(v).argmax(), axis=0, arr=votes)
```

## When Averaging Fails

- If trees are highly correlated, averaging gives little benefit (see 04).
- If a rare class is systematically under-predicted, majority voting can mask
  minority performance → consider class weighting (19).

## Next Steps

- 12 - OOB Score
- 13 - Feature Importance
