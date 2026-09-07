# 09 - Implementing with scikit-learn

The canonical library for Random Forest in Python is scikit-learn.

## Core Classes

- `sklearn.ensemble.RandomForestClassifier`
- `sklearn.ensemble.RandomForestRegressor`

## Minimal Workflow

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

## Other Useful Implementations

- `BaggingClassifier` / `BaggingRegressor` (plain bagging)
- `ExtraTreesClassifier` / `ExtraTreesRegressor` (Extremely Randomized Trees)
- `HistGradientBoostingClassifier` (boosting alternative, fast)

## Common Parameters

```python
RandomForestClassifier(
    n_estimators=100,     # number of trees
    max_depth=None,       # grow until pure / min_samples_split
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,       # enable bootstrap sampling
    oob_score=True,       # track out-of-bag error
    n_jobs=-1,            # parallelize across cores
    random_state=42       # reproducibility
)
```

## Datasets for Practice

- Classification: Iris, Breast Cancer, Digits, Titanic
- Regression: Boston (deprecated), California Housing, Diabetes

## Next Steps

- 10 - Hyperparameter Reference
- 15 - Max Features Tuning
