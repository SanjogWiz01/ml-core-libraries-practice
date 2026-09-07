# 11 - Tree Diversity & Decorrelation

The success of Random Forest hinges on **tree diversity**. This file explains
why and how.

## Why Diversity Matters

If all trees made identical predictions, the ensemble would be no better than a
single tree. Averaging only helps when trees **disagree** (in an unbiased way).

## Sources of Diversity

1. **Bootstrap sampling** (different data per tree) → 03
2. **Random feature subspace** (different features per split) → 04
3. **Model randomness** (random tie-breaking / thresholds) → 14

## Measuring Correlation

Averaging reduces variance most when the average correlation `ρ` between
tree predictions is low:

```
variance(ensemble) = ρ·σ² + (1 - ρ)·σ² / T
```

As `T → ∞`, variance → `ρ·σ²`. So **ρ is the floor** on variance reduction.

## Trade-off Triangle

- Diversity comes at the cost of individual tree accuracy.
- Net gain = variance reduction > slight bias increase.
- `max_features` directly controls this trade-off.

## How to Increase Diversity

- Reduce `max_features`.
- Use `max_samples < 1.0` (subsample the bootstrap).
- Use feature subspacing per split rather than per tree.

## Next Steps

- 14 - Extra Trees vs Random Forest
- 16 - Increasing Estimators
