# 24 - SHAP Values for Random Forest

SHAP (SHapley Additive exPlanations) provides **per-prediction** feature
attributions — a gold standard for model interpretability.

## Background

SHAP attributes each prediction's deviation from the base rate to features,
using cooperative game theory (Shapley values). Guarantees:
- Additivity: attributions sum to the prediction.
- Local accuracy, consistency.

## Why Use it With Random Forest

Random Forest global importance (13/22) tells you *which* features matter but
not *how/what for each row*. SHAP explains **single predictions** and reveals
**direction and interaction**.

## In Python

```python
import shap
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X)          # global overview
shap.force_plot(explainer.expected_value, shap_values[0], X[0])  # single row
shap.dependence_plot(0, shap_values, X)    # interaction view
```

## Key Plots

- **Summary plot**: importance + direction (red = high feature value, blue = low).
- **Force plot**: additive explanation of one prediction.
- **Dependence plot**: how one feature's SHAP changes with another.

## Speed Note

`TreeExplainer` is **fast** for Random Forest (tree-structured). Good for
interactive pipelines.

## Practical Advice (Pareto)

- Use SHAP when you must explain **individual** decisions (loans, medical).
- Pair with PDP (23) for aggregate effect and SHAP for local effect.
- Watch out: SHAP with correlated features needs `feature_perturbation`.

## Next Steps

- 23 - Partial Dependence Plots
- 13 - Feature Importance