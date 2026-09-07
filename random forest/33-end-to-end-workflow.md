# 33 - End-to-End Workflow with Random Forest

The complete, practical pipeline for a data scientist.

## Step 1 — Setup & Split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

## Step 2 — Preprocess (fit inside folds/CV)

- Encode categoricals (20).
- Impute missing (18).
- No scaling needed.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

pre = ColumnTransformer([
    ('num', 'passthrough', numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
])
pipe = Pipeline([('pre', pre), ('rf', RandomForestClassifier(random_state=42))])
```

## Step 3 — Validate & Tune

```python
from sklearn.model_selection import GridSearchCV

grid = {
    'rf__max_features': ['sqrt', 'log2', 0.4],
    'rf__min_samples_leaf': [1, 3, 5],
    'rf__n_estimators': [200],
}
gs = GridSearchCV(pipe, grid, cv=5, scoring='roc_auc', n_jobs=-1)
gs.fit(X_train, y_train)
```

## Step 4 — Evaluate on Test

- Classification metrics (26) / regression metrics (29).
- Confusion matrix, ROC curve, PR curve.
- OOB sanity check (12).

## Step 5 — Explain & Deploy

- Feature importance (13) + SHAP (24).
- Save with `joblib` / `pickle`.
- Monitor drift on calibration (25).

## Pareto Summary (the 20% that does 80%)

1. Solid pipeline with CV.
2. Tune 3–4 knobs.
3. Robust metrics.
4. Deployment + drift checks.

## Next Steps

- 34 - Common Pitfalls
- 35 - Cheat Sheet