# 07 - Random Forest for Regression

Random Forest can predict continuous targets.

## How Predictions are Made

1. Each tree produces a real-valued prediction.
2. The final prediction is the **average** across all trees.

```
y_hat = (1/T) · Σ_tree(tree_prediction)
```

## Split Criterion

Trees split to minimize the **Mean Squared Error (MSE)**:

```
MSE = (1/n) · Σ (y_i - ŷ)²
```

This measures how close the leaf's average is to the actual values.

## Properties

- Piecewise-constant predictions (trees can't extrapolate beyond observed range).
- Good for capturing **non-linear** regression relationships.
- Reliable uncertainty proxy via tree variance (see 21).

## R² as Evaluation

The coefficient of determination:

```
R² = 1 - SS_res / SS_tot
```

`1.0` = perfect fit, `0` = as good as the mean.

## Example Sketch (sklearn)

```python
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(n_estimators=300, random_state=42)
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
```

## Caveats

- Cannot predict outside the training target range.
- `max_features` for regression is often `p/3` or `"sqrt"` — tune it.

## Next Steps

- 08 - Voting & Averaging Details
- 21 - Confidence Intervals & Uncertainty
