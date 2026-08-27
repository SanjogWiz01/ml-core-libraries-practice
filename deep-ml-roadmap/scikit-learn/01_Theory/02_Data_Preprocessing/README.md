# Data Preprocessing

Real data is always messy. Preprocessing converts raw data into something a model can learn from.

---

## The Golden Rule: Split First, Preprocess Second

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit preprocessing ONLY on train, transform both
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

**Never fit on the whole dataset.** That leaks test statistics into training.
The clean way to enforce this: use `Pipeline`.

---

## Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% test
    random_state=42,    # reproducibility
    stratify=y          # for classification: preserve class proportions
)
```

- `test_size=0.2` → 80/20 split
- `stratify=y` → critical for imbalanced classification problems
- `random_state` → same split every run

---

## Missing Values — `SimpleImputer`

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')   # or 'median', 'most_frequent', 'constant'
X_train_imp = imputer.fit_transform(X_train)
X_test_imp  = imputer.transform(X_test)
```

| Strategy | Use when |
|---|---|
| `mean` | Numeric, few outliers |
| `median` | Numeric, outliers present |
| `most_frequent` | Categorical or numeric |
| `constant` | You want to fill with a specific value (e.g., 0 or 'Unknown') |

**Always impute before scaling.**

---

## Feature Scaling

Linear models, KNN, SVM, and neural networks are all sensitive to scale.
Tree-based models (Decision Tree, Random Forest, Gradient Boosting) are **not** — but it doesn't hurt.

### StandardScaler

Transforms each feature to mean=0, std=1.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

- **When**: Most models. Default choice.
- **Formula**: `z = (x - mean) / std`
- **Sensitive to outliers**: Yes (because mean/std are affected)

### MinMaxScaler

Scales each feature to [0, 1].

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
```

- **When**: Neural networks, image pixel values, algorithms that expect bounded input
- **Formula**: `x_scaled = (x - min) / (max - min)`
- **Sensitive to outliers**: Very much — one outlier compresses the rest

### RobustScaler

Uses median and IQR instead of mean/std.

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
```

- **When**: Data with significant outliers
- **Formula**: `x_scaled = (x - median) / IQR`
- **Sensitive to outliers**: No — that's its whole point

### When to skip scaling

- Decision Tree, Random Forest, Gradient Boosting → tree splits don't care about scale
- Target variable for regression → usually don't scale y

---

## Categorical Encoding

ML models require numeric input. Categorical features must be encoded.

### OneHotEncoder — for nominal categories

```python
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
enc.fit_transform(X_train[['color']])
# 'red' → [1, 0, 0], 'blue' → [0, 1, 0], 'green' → [0, 0, 1]
```

- **Use for**: categories with NO natural order (color, city, brand)
- **Warning**: creates one column per category — can explode with high cardinality
- `handle_unknown='ignore'` → safe when test has unseen categories

### OrdinalEncoder — for ordinal categories

```python
from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder(categories=[['low', 'medium', 'high']])
enc.fit_transform(X_train[['education']])
# 'low' → 0, 'medium' → 1, 'high' → 2
```

- **Use for**: categories WITH a natural order (size: S/M/L, rating: 1-5)
- Provide explicit `categories` list to control order

---

## Feature Engineering

Creating new features from existing ones to give the model more signal.

```python
# Ratio feature
df['price_per_sqft'] = df['price'] / df['sqft']

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Binning
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100],
                          labels=['child', 'young', 'adult', 'senior'])
```

Feature engineering is often the highest-leverage step in a real ML project.

---

## Pipeline

Chains preprocessing steps and a model into one object. Prevents leakage.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('imputer',  SimpleImputer(strategy='median')),
    ('scaler',   StandardScaler()),
    ('model',    LogisticRegression())
])

pipe.fit(X_train, y_train)    # all steps fit on train only
pipe.predict(X_test)           # all steps transform test consistently
pipe.score(X_test, y_test)
```

With a Pipeline you can:
- Cross-validate the whole workflow: `cross_val_score(pipe, X, y)`
- Tune any step's params: `GridSearchCV(pipe, {'model__C': [0.1, 1, 10]})`
- Save/load the whole pipeline: `joblib.dump(pipe, 'model.joblib')`

---

## ColumnTransformer — Mixed Data Types

Real datasets have both numeric and categorical columns. `ColumnTransformer` applies different transformers to each column group.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

numeric_features = ['age', 'income', 'balance']
categorical_features = ['education', 'job', 'marital']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer,  numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model',        RandomForestClassifier())
])

full_pipeline.fit(X_train, y_train)
```

This is the **production-ready preprocessing pattern**. Use it for every real project.

---

## Preprocessing Decision Flowchart

```
For each column:
    Is it numeric?
        → Are there missing values?    → SimpleImputer(strategy='median')
        → Is the scale important?      → StandardScaler (default) or RobustScaler (outliers)
    
    Is it categorical?
        → Does the order matter?
            Yes → OrdinalEncoder (provide categories list)
            No  → OneHotEncoder (handle_unknown='ignore')
        → Are there missing values?    → SimpleImputer(strategy='most_frequent')
```
