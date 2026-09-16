# 16 - Choosing the Number of Trees

`n_estimators` controls ensemble size. It's important to get this right.

## The Behavior

- Adding trees always **reduces variance** (up to a point).
- Accuracy improves rapidly at first, then **plateaus** (diminishing returns).
- More trees = more compute and memory.

## The Plateau (Pareto Insight)

Most of the benefit comes from the **first few hundred** trees. Adding 1000 →
2000 trees yields almost no gain but doubles cost. Typically:

```
100 trees   -> good baseline
200-500     -> most of the benefit
1000+       -> marginal gains, mainly stability
```

## How to Choose

- Plot OOB or CV score vs `n_estimators` and find the elbow.
- Use a high enough value that results are **stable** run-to-run.
- Monitor memory usage (trees are stored in memory).

```python
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

scores = []
for n in [50, 100, 200, 400, 800]:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    s = cross_val_score(rf, X, y, cv=5).mean()
    scores.append(s)
plt.plot([50,100,200,400,800], scores)
```

## Balance

Pick the smallest `n_estimators` that reaches the plateau — it saves compute
with negligible loss.

## Next Steps

- 17 - Depth & Leaf Constraints
- 18 - Imputation & Missing Values
