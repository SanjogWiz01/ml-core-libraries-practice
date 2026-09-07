# 18 - Missing Values & Imputation

How to handle missing data with Random Forest.

## Random Forest and Missing Data

- **sklearn's RandomForest cannot accept NaN** directly.
- Two good options: impute before, or use tree-based imputation.

## Option 1: Simple Imputation

```python
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy='median')   # or 'mean', 'most_frequent'
X_imputed = imp.fit_transform(X)
```

- Fast, works well for Random Forest since it cares about ordering only.
- Caveat: can inject bias.

## Option 2: Iterative / MICE Imputation

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imp = IterativeImputer(random_state=42)
X_imputed = imp.fit_transform(X)
```

- Models each feature as a function of others.
- More accurate but slower.

## Option 3: Add Missingness Indicator

Create a binary flag column per feature indicating whether it was missing.
This lets the model learn that "missing" itself is informative.

## Proximity-Based Handling (advanced)

Random Forest can compute **proximity** between samples sharing leaves, which
can be used for imputation (fill missing values with weighted opinions of
neighbors). Nice for research-grade pipelines.

## Practical Advice (Pareto)

For most tabular tasks:
1. Median imputation is a fast, solid default.
2. Add missingness indicators if missing rate is informative.
3. Upgrade to IterativeImputer only when data is large and quality matters.

## Next Steps

- 19 - Class Weighting & Imbalance
- 20 - Categorical & Scaling
