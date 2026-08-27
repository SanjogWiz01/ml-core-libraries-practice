# Model Evaluation & Optimization

A model is only as good as your ability to measure it correctly.

---

## REGRESSION METRICS

Given: `y_true = [3, 5, 2, 8]`, `y_pred = [2.5, 5.5, 2, 7]`

### MAE — Mean Absolute Error

```
MAE = mean(|y_true - y_pred|)
```

- Same units as the target
- Robust to outliers
- Easy to interpret: "on average, my predictions are off by X"

```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
```

### MSE — Mean Squared Error

```
MSE = mean((y_true - y_pred)²)
```

- Penalizes large errors heavily (squared)
- Not in original units (use RMSE for interpretation)
- Differentiable — used in optimization

```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)
```

### RMSE — Root Mean Squared Error

```
RMSE = sqrt(MSE)
```

- Same units as the target
- More sensitive to outliers than MAE
- Industry standard for regression

```python
import numpy as np
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# or
rmse = mean_squared_error(y_true, y_pred, squared=False)  # sklearn >= 1.4
```

### R² — Coefficient of Determination

```
R² = 1 - SS_res / SS_tot
   = 1 - Σ(y_true - y_pred)² / Σ(y_true - mean(y_true))²
```

- Range: (-∞, 1.0]. 1.0 is perfect. 0.0 = as good as predicting the mean.
- Negative means *worse* than predicting the mean (your model is bad)
- Interpretation: "the model explains X% of the variance in y"
- **Caution**: R² always increases as you add features, even useless ones → use adjusted R²

```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
```

### Which to use?

- **MAE** when outliers shouldn't dominate and you want interpretability
- **RMSE** when large errors are especially bad (default for most regression tasks)
- **R²** for communicating model quality to stakeholders

---

## CLASSIFICATION METRICS

Binary classification example:
```
y_true = [1, 0, 1, 1, 0, 1, 0]
y_pred = [1, 0, 0, 1, 0, 1, 1]
```

### Confusion Matrix

```
                   Predicted
                  Neg    Pos
Actual  Neg    [ TN=2   FP=1 ]
        Pos    [ FN=1   TP=3 ]

TP = True Positive  (predicted positive, actually positive)
TN = True Negative  (predicted negative, actually negative)
FP = False Positive (predicted positive, actually negative) — Type I Error
FN = False Negative (predicted negative, actually positive) — Type II Error
```

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
```

### Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

- **Misleading with imbalanced classes**: 99% accuracy if you always predict "No" when 99% are negative
- Use as a quick sanity check only

### Precision

```
Precision = TP / (TP + FP)
```

- Of everything I predicted positive, what fraction was actually positive?
- **Use when FP is costly**: spam filter (marking real email as spam), innocent person conviction

### Recall (Sensitivity, TPR)

```
Recall = TP / (TP + FN)
```

- Of everything that's actually positive, what fraction did I catch?
- **Use when FN is costly**: cancer diagnosis (missing cancer is worse than false alarm), fraud detection

### F1 Score

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

- Harmonic mean of precision and recall
- Best single metric when both matter equally
- **Use for imbalanced datasets**

### Precision-Recall Tradeoff

- Lowering the classification threshold → more positives predicted → Recall ↑, Precision ↓
- Raising the threshold → fewer positives predicted → Precision ↑, Recall ↓
- Tune the threshold based on your cost structure

### ROC-AUC

ROC = Receiver Operating Characteristic curve  
AUC = Area Under the Curve (ROC)

```
AUC = 1.0  → perfect classifier
AUC = 0.5  → random classifier (no information)
AUC < 0.5  → worse than random (flip predictions)
```

- Measures how well the model ranks positives above negatives
- **Threshold-independent** — evaluates performance across all possible thresholds
- Best general-purpose metric for binary classification

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

print(classification_report(y_true, y_pred))  # precision, recall, F1 in one call
auc = roc_auc_score(y_true, model.predict_proba(X_test)[:, 1])
```

### Which classification metric to use?

| Situation | Use |
|---|---|
| Balanced classes | Accuracy, F1 |
| Imbalanced classes | F1, ROC-AUC, Precision-Recall AUC |
| FP cost is high | Precision |
| FN cost is high | Recall |
| Need ranking/probability | ROC-AUC |

---

## CROSS-VALIDATION

A single train/test split gives you one estimate of performance — which depends on which samples ended up in the test set by random chance.

Cross-validation gives you **k independent estimates** and their variance.

### K-Fold

```
Data: [1  2  3  4  5  6  7  8  9  10]

Fold 1: test=[1,2]   train=[3,4,5,6,7,8,9,10]
Fold 2: test=[3,4]   train=[1,2,5,6,7,8,9,10]
Fold 3: test=[5,6]   train=[1,2,3,4,7,8,9,10]
Fold 4: test=[7,8]   train=[1,2,3,4,5,6,9,10]
Fold 5: test=[9,10]  train=[1,2,3,4,5,6,7,8]

CV score = mean([score_1, score_2, score_3, score_4, score_5])
```

```python
from sklearn.model_selection import cross_val_score, KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
print(f"CV R²: {scores.mean():.3f} ± {scores.std():.3f}")
```

### Stratified K-Fold

Preserves class proportions in each fold — **always use for classification**.

```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
```

### Common scoring strings

```python
# Regression
'r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error'

# Classification
'accuracy', 'f1', 'precision', 'recall', 'roc_auc'
# for multi-class: 'f1_macro', 'f1_weighted'
```

Note: sklearn uses negative scores for metrics where lower=better (so `maximize_score` works universally).

---

## HYPERPARAMETER TUNING

Hyperparameters are settings you choose before training (not learned from data).

### GridSearchCV — Exhaustive search

Tests every combination in the grid. Guaranteed to find the best in the grid.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5, None],
    'model__min_samples_leaf': [1, 5, 10]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,       # use all CPU cores
    verbose=1
)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)  # CV score
best_model = grid_search.best_estimator_
```

**Cost**: `len(param_grid values) × n_splits` model fits. With 3×3×3 grid and 5-fold CV = 135 fits.

### RandomizedSearchCV — Random sampling

Samples `n_iter` random combinations. More efficient when grid is large.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'model__n_estimators': randint(50, 500),
    'model__max_depth': randint(3, 20),
    'model__min_samples_leaf': randint(1, 20),
    'model__learning_rate': uniform(0.01, 0.2)
}

rand_search = RandomizedSearchCV(
    pipeline,
    param_dist,
    n_iter=50,        # 50 random combinations
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42
)
rand_search.fit(X_train, y_train)
```

**Rule**: Use GridSearch when you have ≤3 hyperparameters with few values. Use RandomizedSearch otherwise.

---

## MODEL SELECTION WORKFLOW

```
1. Establish a baseline (always predict mean for regression / majority class for classification)
2. Try LinearRegression / LogisticRegression → fast, interpretable
3. Try RandomForest → robust, often better
4. Try GradientBoosting → usually best, but needs tuning
5. Cross-validate each with same CV splits
6. Select best model family
7. Tune hyperparameters with RandomizedSearchCV → GridSearchCV (narrow range)
8. Evaluate on test set (ONCE)
```

**Never tune on the test set.** If you look at test results and then re-tune, you've created leakage.
