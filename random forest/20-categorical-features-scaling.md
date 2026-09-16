# 20 - Categorical Features & Scaling

Random Forest handles categorical and numeric features differently from many
algorithms.

## Scaling Is NOT Needed

Random Forest is **scale-invariant** — it only compares feature values to
thresholds. No standardization / normalization required.

```
No need for StandardScaler / MinMaxScaler!
```

This is a major convenience vs SVM, KNN, and neural nets.

## Categorical Features

sklearn's Random Forest historically required **numeric** input — categoricals
must be encoded:

### One-Hot Encoding
```python
from sklearn.preprocessing import OneHotEncoder
```
- Creates binary columns per category.
- Can inflate feature space with high-cardinality categories.

### Ordinal Encoding
- Number each category 1..k. Works because trees only need order for splits.

### Native Categorical Support
Histogram-based boosters (LightGBM, CatBoost) handle categoricals natively;
Random Forest in sklearn does not. For native support, consider
`sklearn.ensemble.HistGradientBoostingClassifier` (actually a booster).

## Encoding Advice (Pareto)

- For high-cardinality categoricals, prefer **ordinal** or target encoding over
  one-hot (keeps tree splits meaningful and avoids many sparse columns).
- Random Forest tolerates ordinal-encoded categoricals well since splits are
  threshold-based.

## Data Types Summary

| Data | Random Forest Requirement |
|------|---------------------------|
| Numeric | Use as-is (no scaling) |
| Categorical | Encode to numeric |
| Missing | Impute or flag (see 18) |

## Next Steps

- 18 - Missing Values
- 28 - Feature Selection Workflow
