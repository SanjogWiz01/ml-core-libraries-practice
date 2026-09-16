# 23 - Partial Dependence Plots (PDPs)

PDPs and ICE curves make Random Forest explainable.

## What They Show

How the prediction changes as a feature varies, **averaging out** the effect of
all other features:

```
PDP(x_j) = (1/n) · Σ_i f(x_j, x_i_others)
```

## Types of Plots

### Partial Dependence Plot (PDP)
- Average predicted output vs one (or two) features.
- Shows the **marginal** effect.
- Can hide heterogeneity.

### Individual Conditional Expectation (ICE)
- One curve per sample.
- Reveals **heterogeneous** effects the average hides.

## In sklearn

```python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    rf, X_train, features=['age', 'income'],
    kind='both'  # plots PDP + ICE
)
```

## Interpretation Tips

- 1-D PDP: does the feature raise or lower predictions across ranges?
- 2-D PDP: interaction surfaces — how do two features jointly affect prediction?
- Be careful: PDP averages can mislead when features are highly correlated.

## When to Use (Pareto)

- Explaining a single important feature (from 13/22) to stakeholders.
- Checking monotonicity / direction — e.g., is `credit_score` positively
  relating to `loan_approval_probability`?

## Next Steps

- 24 - SHAP Values
- 13 - Feature Importance