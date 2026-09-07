# 31 - Random Forest vs Other Models

Where Random Forest stands against linear models, SVMs, and k-NN.

## Linear Models (Logistic / Linear Regression)

| Aspect | Random Forest | Linear |
|--------|---------------|--------|
| Assumptions | None | Linearity |
| Interactions | Captured automatically | Must engineer |
| Interpretability | Moderate | High |
| Needed data | More | Fine with less |
| Extrapolation | Poor | Good |

**Take**: use linear when domain demands coefficients & extrapolation; RF when
relationships are nonlinear.

## Support Vector Machines

| Aspect | Random Forest | SVM |
|--------|---------------|-----|
| Scaling needed | No | Yes |
| Kernel tricks | Not needed | RBF/poly kernels |
| Training data | Large OK | Slow to scale |
| High dimensionality | Handles OK | Very strong |
| Tuning | Few params | Sensitive to C, gamma |

**Take**: RF is easier to use end-to-end; SVM can edge out on small,
high-dimensional, well-scaled data.

## k-Nearest Neighbors

| Aspect | Random Forest | k-NN |
|--------|---------------|------|
| Training time | Moderate | ~Zero |
| Prediction time | Fast | Slow (per query) |
| Feature scaling | Not needed | Essential |
| Curse of dim. | Less affected | Strongly affected |

**Take**: RF wins on moderate/large tabular data; k-NN shines in
low-dimensional lazily-indexed settings.

## Rules of Thumb (Pareto 20/80)

- RF is a **top default** for structured tabular data.
- Spend the 20% effort on: tune `max_features`, `min_samples_leaf`, trees count;
  then compare vs boosting (30).

## Next Steps

- 30 - vs Gradient Boosting
- 05 - Ensemble & Bagging