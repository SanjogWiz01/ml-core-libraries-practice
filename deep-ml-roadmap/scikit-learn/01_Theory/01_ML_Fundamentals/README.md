# ML Fundamentals

The mental model you need before writing a single line of scikit-learn.

---

## What Is Machine Learning?

ML is the practice of writing programs that **learn patterns from data** rather than following explicit rules.

Instead of:
```
if age > 60 and cholesterol > 240: high_risk = True
```

You provide examples (age, cholesterol, outcome) and the algorithm learns the rule itself.

---

## Three Types of ML

| Type | Definition | Example |
|---|---|---|
| **Supervised** | Labeled data — you know the answer | Predict house price, classify spam |
| **Unsupervised** | No labels — find hidden structure | Customer segments, anomaly detection |
| **Semi-supervised** | Mix of labeled and unlabeled | Large unlabeled + small labeled corpus |

This repo focuses on **supervised** (majority) and **unsupervised** (clustering, PCA).

---

## Supervised Learning: Two Subtypes

### Regression
- Target is **continuous** (a number)
- Examples: house price, temperature, salary
- Metrics: MAE, MSE, RMSE, R²

### Classification
- Target is **discrete** (a category)
- Examples: spam/not-spam, cancer/benign, 0-9 digit
- Metrics: accuracy, precision, recall, F1, ROC-AUC

---

## Key Vocabulary

### Features and Target

```
Feature matrix X → the inputs (what you know)
Target vector  y → the output (what you're predicting)

          age  height  weight   ← features (columns)
X =  [   25,    170,    65   ]  ← one sample (row)
     [   34,    180,    80   ]
     ...
y =  [  72000, 88000, ... ]     ← salaries (regression)
     or
y =  [  0, 1, 1, 0, ... ]      ← binary labels (classification)
```

### Train / Validation / Test Split

```
All data
    ├── Training set   (~70%)  → model learns from this
    ├── Validation set (~15%)  → tune hyperparameters, compare models
    └── Test set       (~15%)  → final, one-time evaluation only
```

**The test set is sacred.** Look at it once, at the very end. Peeking earlier gives you an over-optimistic estimate of real performance.

With scikit-learn:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## Overfitting and Underfitting

```
                    High Bias              Good fit          High Variance
                  (Underfitting)                             (Overfitting)
                       |                    |                    |
Training error:      HIGH                 LOW                  LOW
Test error:          HIGH                 LOW                  HIGH
```

### Underfitting
- Model is too simple to capture real patterns
- Signs: high error on *both* train and test
- Fix: more features, more complex model, less regularization

### Overfitting
- Model memorizes training data, fails to generalize
- Signs: low train error, high test error
- Fix: regularization, less complexity, more data, dropout, early stopping

---

## Bias-Variance Tradeoff

Every model's test error = **Bias² + Variance + Irreducible Noise**

| Term | Meaning | Cause |
|---|---|---|
| **Bias** | Systematic error — model too rigid | Too simple |
| **Variance** | Sensitivity to training data — wiggly | Too complex |
| **Irreducible** | Random noise in the problem itself | Cannot eliminate |

- High bias → underfitting → simple models (Linear Regression)
- High variance → overfitting → complex models (deep Decision Trees)
- Goal: find the **sweet spot** — cross-validation is how you find it

---

## Data Leakage

**The silent model-killer.** When information from the test set (or future) leaks into training.

### Why it's dangerous
Leakage produces artificially high training/CV scores. The model looks great but fails in production because it relied on information it wouldn't have in the real world.

### Common sources

1. **Preprocessing before split**
   ```python
   # WRONG — scaler sees test data
   X_scaled = scaler.fit_transform(X)
   X_train, X_test = train_test_split(X_scaled, ...)

   # RIGHT — scaler only learns from train
   X_train, X_test = train_test_split(X, ...)
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled  = scaler.transform(X_test)
   ```

2. **Target encoding with all data**
   Using the target to engineer features before splitting.

3. **Using future data**
   In time-series: using tomorrow's data to predict today.

### The solution: `Pipeline`
```python
from sklearn.pipeline import Pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])
pipe.fit(X_train, y_train)   # scaler fits on train only
pipe.predict(X_test)          # scaler applies train stats to test
```

Pipelines eliminate leakage in preprocessing automatically.

---

## The ML Workflow (High Level)

```
1. Define the problem        → regression or classification? What's success?
2. Get and explore data      → shape, types, distributions, missing values
3. Clean                     → handle nulls, fix types, remove duplicates
4. Split                     → train/test FIRST, before any preprocessing
5. Preprocess (train only)   → impute, scale, encode via Pipeline
6. Train model               → fit on training data
7. Evaluate                  → CV score → test score (once)
8. Tune                      → GridSearchCV or RandomizedSearchCV
9. Final evaluation          → test set, metrics, confusion matrix
10. Save model               → joblib.dump()
```

Every step in this repo maps to one of these stages.
