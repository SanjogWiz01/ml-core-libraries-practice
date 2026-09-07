# 15 - Tuning Max Features

`max_features` is the single most impactful hyperparameter for Random Forest — this is the 20% that drives 80% of tuning gains.

## What It Controls

The number of features considered at **each split**. The higher it is, the more
correlated the trees (see 11).

## Typical Values

| Task | Heuristic | Notes |
|------|-----------|-------|
| Classification | `sqrt(p)` | Default |
| Classification | `log2(p)` | Fewer, more random |
| Regression | `p/3` | Common heuristic |
| Any | small int like 1–5 | Aggressive randomization |

## Effect on Performance

- Too small → trees too degraded (high bias).
- Too large → trees too correlated (little variance reduction).
- Best value usually lands between `sqrt(p)` and `p/3`.

## How to Tune

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7]}
gs = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
gs.fit(X_train, y_train)
print(gs.best_params_)
```

## Practical Advice (Pareto)

- Tune `max_features` **first** among tree-structure knobs.
- Keep `n_estimators` reasonably high and decoupled.
- Watch OOB score as `max_features` changes (see 12).

## Next Steps

- 16 - Number of Trees
- 17 - Depth & Leaf Constraints
