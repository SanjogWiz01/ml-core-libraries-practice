# 30 - Random Forest vs Gradient Boosting

Two ensemble giants — understand when each wins.

## Comparison

| Aspect | Random Forest | Gradient Boosting (XGBoost/LGBM) |
|--------|---------------|----------------------------------|
| Training | Parallel (independent trees) | Sequential (trees correct errors) |
| Bias | Higher | Lower |
| Variance | Lower (averaging) | Higher (needs careful tuning) |
| Speed | Fast to train | Slower to train, fast to predict |
| Tuning | Few params | Many, sensitive params |
| Overfitting | Very robust | Risks overfitting |
| Interpretability | Good | Good (with feature importance) |
| Small data | Great | Can overfit |
| Large data | Good | Often best-in-class |

## When to Choose Random Forest

- Tabular data with limited sample size.
- Need robustness with minimal tuning.
- Must train fast / in parallel.

## When to Choose Gradient Boosting

- Large structured datasets, ranking / kaggle-like benchmarks.
- Careful hyperparameter tuning available.
- Want highest accuracy with regularization.

## Hybrid Advice (Pareto)

Try Random Forest first as a strong, quick baseline. Reach for boosting only
when RF plateaus and you have budget and data.

## Related Algorithms

- Bagging (05) — foundation
- Extra Trees (14) — cheaper cousin
- XGBoost / LightGBM / CatBoost — boosting cousins

## Next Steps

- 31 - Random Forest vs Other Models
- 05 - Ensemble & Bagging