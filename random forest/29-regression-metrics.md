# 29 - Regression Metrics for Random Forest

Evaluating Random Forest regressors correctly.

## Core Metrics

### Mean Squared Error (MSE)
`Σ(y - ŷ)² / n`. Heavily penalizes large errors. Sensitive to outliers.

### Root MSE (RMSE)
`sqrt(MSE)` — same units as the target, easy to interpret.

### Mean Absolute Error (MAE)
`Σ|y - ŷ| / n`. Robust to outliers.

### R² (Coefficient of Determination)
`1 - SS_res/SS_tot` — proportion of variance explained.

## In sklearn

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)  # sklearn>=1.0
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
```

## Choosing the Metric (Pareto)

| Situation | Preferred Metric |
|-----------|------------------|
| General baseline | RMSE + R² |
| Outliers present | MAE |
| Business cost ∝ error² | MSE |
| Explainability | RMSE (same units) |

## Watch Out

- R² can be very negative on extrapolation → sign of poor generalization.
- Compare R² against a trivial model (predicting the mean) for sanity.

## Next Steps

- 07 - Regression
- 27 - Cross-Validation