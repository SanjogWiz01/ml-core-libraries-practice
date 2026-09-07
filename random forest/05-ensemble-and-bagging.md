# 05 - Ensemble Learning & Bagging

This file places Random Forest within the broader family of ensemble methods.

## What is an Ensemble?

Combining the predictions of many models to produce a single, stronger
prediction. The goal is to reduce error beyond any single model.

## The Bias-Variance Trade-off

- **Bias**: error from simplistic assumptions.
- **Variance**: error from sensitivity to training data.
- Ensembles reduce variance (and sometimes bias) by combining models.

## Two Main Families

### 1. Averaging / Parallel (Bagging family)
- Train models independently (in parallel) on different data subsets.
- Average / vote their predictions.
- Reduces **variance**.
- Examples: Bagging, **Random Forest**, Extra Trees.

### 2. Boosting (Sequential)
- Train models sequentially, each correcting the previous errors.
- Reduces **bias**.
- Examples: AdaBoost, Gradient Boosting, XGBoost.

## Bagging vs Random Forest

| Aspect | Bagging | Random Forest |
|--------|---------|---------------|
| Data sampling | Bootstrap | Bootstrap |
| Feature sampling | All features | Random subset |
| Tree decorrelation | Moderate | High |
| Typical accuracy | Good | Better |

## Why Averaging Helps (Law of Large Numbers)

If each tree has error variance `σ²` and trees are independent, the variance of
the average is `σ²/T`. In practice trees are correlated by `ρ`, giving:

```
variance(average) = ρ·σ² + (1-ρ)·σ²/T
```

Reducing `ρ` (decorrelation) is why Random Forest beats plain bagging.

## Next Steps

- 06 - Classification
- 07 - Regression
