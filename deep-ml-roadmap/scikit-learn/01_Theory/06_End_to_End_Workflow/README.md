# End-to-End ML Workflow

The complete pipeline from raw data to deployed model.

---

## Overview

```
Problem Definition
    → Get Data
    → EDA
    → Cleaning
    → Train/Test Split        ← split FIRST
    → Preprocessing           ← fit ONLY on train
    → Feature Engineering
    → Pipeline + ColumnTransformer
    → Train Model
    → Cross-Validation
    → Hyperparameter Tuning
    → Final Evaluation
    → Save Model (joblib)
    → Load Model
    → Predict on New Data
```

---

## Step 1: Problem Definition

Before touching data, answer:
- Is this regression (continuous target) or classification (categorical)?
- What does success look like? (R² > 0.85? F1 > 0.80?)
- What are the costs of different errors? (FP vs FN tradeoff)
- What's the prediction latency requirement?

---

## Step 2: Exploratory Data Analysis (EDA)

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/raw/dataset.csv')

# Shape and types
print(df.shape)
print(df.dtypes)
print(df.info())

# Missing values
print(df.isnull().sum())
print(df.isnull().mean().sort_values(ascending=False))  # fraction missing

# Target distribution
df['target'].hist(bins=30)                # regression
df['target'].value_counts().plot.bar()    # classification

# Feature distributions
df.describe()

# Correlations
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f')

# Pairplot for small datasets
sns.pairplot(df, hue='target')
```

**EDA goal**: understand shape, distributions, missingness, outliers, correlations, and target balance.

---

## Step 3: Cleaning

```python
# Drop completely useless columns
df = df.drop(columns=['id', 'timestamp_raw'])

# Fix dtypes
df['age'] = df['age'].astype(int)

# Remove exact duplicates
df = df.drop_duplicates()

# Clip extreme outliers (optional — verify they're errors, not valid extremes)
df['income'] = df['income'].clip(upper=df['income'].quantile(0.99))
```

Do not remove missing values here — let `SimpleImputer` inside the Pipeline handle them.

---

## Step 4: Train/Test Split

**Do this FIRST. Before any preprocessing.**

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y if is_classification else None
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

---

## Step 5–7: Preprocessing + Feature Engineering + Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Identify column types
numeric_cols = X_train.select_dtypes(include='number').columns.tolist()
categorical_cols = X_train.select_dtypes(include='object').columns.tolist()

# Feature engineering (do this BEFORE defining transformers)
# Example: add a derived feature
X_train = X_train.copy()
X_test = X_test.copy()
X_train['rooms_per_person'] = X_train['rooms'] / X_train['population'].clip(lower=1)
X_test['rooms_per_person']  = X_test['rooms'] / X_test['population'].clip(lower=1)
numeric_cols.append('rooms_per_person')

# Transformers
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer,  numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])
```

---

## Step 8: Train Model

```python
from sklearn.ensemble import RandomForestRegressor

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

full_pipeline.fit(X_train, y_train)
```

---

## Step 9: Cross-Validation

Evaluate the full pipeline, not just the model:

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # for classification
# cv = KFold(n_splits=5, shuffle=True, random_state=42)          # for regression

cv_scores = cross_val_score(
    full_pipeline, X_train, y_train,
    cv=cv,
    scoring='r2',   # or 'f1', 'roc_auc', etc.
    n_jobs=-1
)
print(f"CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

---

## Step 10: Hyperparameter Tuning

Use `RandomizedSearchCV` first for broad search, then `GridSearchCV` to narrow:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'model__n_estimators': randint(100, 500),
    'model__max_depth': [None, 5, 10, 20],
    'model__min_samples_leaf': randint(1, 20),
}

search = RandomizedSearchCV(
    full_pipeline, param_dist,
    n_iter=30, cv=5, scoring='r2',
    n_jobs=-1, random_state=42, verbose=1
)
search.fit(X_train, y_train)

print(f"Best CV score: {search.best_score_:.4f}")
print(f"Best params: {search.best_params_}")

best_pipeline = search.best_estimator_
```

---

## Step 11: Final Evaluation on Test Set

Do this ONCE. The test set was never used until now.

```python
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

y_pred = best_pipeline.predict(X_test)

print(f"Test MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"Test R²:   {r2_score(y_test, y_pred):.4f}")
```

If test score is much lower than CV score → overfitting or leakage. Investigate.

---

## Step 12: Save Model with joblib

```python
import joblib
from pathlib import Path

models_dir = Path(__file__).parent.parent / 'models'
models_dir.mkdir(exist_ok=True)

joblib.dump(best_pipeline, models_dir / 'house_price_model.joblib')
print("Model saved.")
```

`joblib` is faster than `pickle` for scikit-learn models with large arrays.

---

## Step 13: Load Model and Predict

```python
loaded_pipeline = joblib.load(models_dir / 'house_price_model.joblib')

# New data — same format as original features, before preprocessing
new_data = pd.DataFrame({
    'longitude': [-122.23],
    'latitude': [37.88],
    'housing_median_age': [41],
    'total_rooms': [880],
    'total_bedrooms': [129],
    'population': [322],
    'households': [126],
    'median_income': [8.3252],
    'ocean_proximity': ['NEAR BAY']
})

prediction = loaded_pipeline.predict(new_data)
print(f"Predicted house value: ${prediction[0]:,.0f}")
```

The pipeline applies all preprocessing automatically — no need to manually scale or encode new data.

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Preprocessing before split | Train/test scores too close, real-world performance worse | Always split first |
| Fitting scaler on test | Slightly inflated test score | Use Pipeline |
| Tuning on test set | Test score looks great but model fails in production | Use CV for tuning |
| Missing `random_state` | Results differ every run | Set `random_state=42` everywhere |
| Wrong metric | Accuracy 97% but F1 0.4 (imbalanced) | Match metric to problem type |
| Single train/test split | High variance in performance estimate | Use cross-validation |
