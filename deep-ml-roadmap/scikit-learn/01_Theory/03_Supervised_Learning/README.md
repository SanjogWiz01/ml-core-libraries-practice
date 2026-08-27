# Supervised Learning

All algorithms follow the same interface in scikit-learn:

```python
model.fit(X_train, y_train)      # learn from data
model.predict(X_test)             # output predictions
model.score(X_test, y_test)       # default metric (R² for regression, accuracy for classification)
```

---

## REGRESSION ALGORITHMS

### Linear Regression

**What**: Fits a line (or hyperplane) through the data by minimizing squared errors.
**Formula**: `y = w₀ + w₁x₁ + w₂x₂ + ...`

**When to use**:
- Baseline for any regression problem
- Interpretability matters (coefficients = feature importance)
- Relationship between features and target is roughly linear

**Advantages**: Fast, interpretable, works well when linear
**Disadvantages**: Fails on non-linear relationships, sensitive to outliers, no built-in regularization

**Key hyperparameters**: None for plain LinearRegression

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
```

---

### Ridge Regression (L2)

**What**: Linear Regression + L2 penalty on coefficients. Shrinks all coefficients toward zero.
**Formula**: `Loss = MSE + α * Σwᵢ²`

**When to use**:
- Collinear features (correlated predictors)
- Want to keep all features but shrink their influence

**Advantages**: Handles multicollinearity, never zeroes out features
**Disadvantages**: Can't do feature selection (no zeros)

**Key hyperparameters**:
- `alpha` — regularization strength. Larger → more shrinkage. Tune with CV.

```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
```

---

### Lasso Regression (L1)

**What**: Linear Regression + L1 penalty. Can shrink coefficients all the way to zero.
**Formula**: `Loss = MSE + α * Σ|wᵢ|`

**When to use**:
- You suspect only a few features are actually relevant
- Automatic feature selection is desirable

**Advantages**: Built-in feature selection (sparse coefficients)
**Disadvantages**: Arbitrary in which features to zero when correlated

**Key hyperparameters**:
- `alpha` — regularization strength

```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)
```

---

### ElasticNet (L1 + L2)

**What**: Combines Ridge and Lasso penalties.
**Formula**: `Loss = MSE + α * (l1_ratio * L1 + (1 - l1_ratio) * L2)`

**When to use**:
- You want both feature selection AND handling of correlated features
- Lasso is too aggressive (zeros out too many)

**Key hyperparameters**:
- `alpha` — overall regularization strength
- `l1_ratio` — mix between L1 and L2 (0=Ridge, 1=Lasso)

```python
from sklearn.linear_model import ElasticNet
model = ElasticNet(alpha=0.1, l1_ratio=0.5)
```

---

### Decision Tree Regressor

**What**: Recursively splits data into regions, predicts the mean of each region.

**When to use**:
- Non-linear relationships
- Interaction effects between features
- Interpretability of decision rules needed

**Advantages**: No scaling needed, handles non-linearity, interpretable
**Disadvantages**: High variance, overfits easily

**Key hyperparameters**:
- `max_depth` — max tree depth (most important — start with 3-5)
- `min_samples_leaf` — min samples in a leaf (prevents tiny leaf nodes)
- `min_samples_split` — min samples to attempt a split

```python
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10)
```

---

### Random Forest Regressor

**What**: Ensemble of Decision Trees trained on random data subsets + random feature subsets. Averages their predictions.

**When to use**:
- Non-linear relationships
- Don't want to tune much (robust defaults)
- Feature importance matters

**Advantages**: Reduces overfitting vs single tree, built-in feature importance, handles missing (somewhat), robust
**Disadvantages**: Slower to train and predict, less interpretable than single tree

**Key hyperparameters**:
- `n_estimators` — number of trees (more=better but slower; 100-500 is typical)
- `max_depth` — depth of each tree (None=fully grown)
- `max_features` — features considered per split ('sqrt' for classification, 1.0 for regression)
- `min_samples_leaf` — minimum samples per leaf

```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
```

---

### Gradient Boosting Regressor

**What**: Builds trees sequentially, each correcting the errors of the previous. Additive model.

**When to use**:
- Best performance on tabular data (along with XGBoost)
- Willing to tune carefully

**Advantages**: Often the most accurate on tabular data, flexible loss functions
**Disadvantages**: Slow to train, many hyperparameters, can overfit

**Key hyperparameters**:
- `n_estimators` — number of trees (use with early stopping)
- `learning_rate` — step size shrinkage (smaller=slower but safer; 0.01-0.1)
- `max_depth` — depth of each tree (3-5 typical)
- `subsample` — fraction of samples per tree (0.8 adds stochasticity)

```python
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4)
```

---

## CLASSIFICATION ALGORITHMS

### Logistic Regression

**What**: Regression model outputting probabilities via the sigmoid function. Despite the name, it's a classifier.
**Formula**: `P(y=1) = sigmoid(w·x) = 1 / (1 + e^(-w·x))`

**When to use**:
- Binary or multi-class classification
- Interpretability needed (log-odds coefficients)
- Fast baseline for classification

**Advantages**: Probabilistic output, fast, works well on linearly separable data, built-in L2 reg
**Disadvantages**: Can't handle non-linear boundaries without feature engineering

**Key hyperparameters**:
- `C` — inverse regularization strength (smaller C = more regularization)
- `penalty` — 'l2' (default), 'l1', 'elasticnet', 'none'
- `solver` — optimization algorithm ('lbfgs' for most; 'saga' for L1)
- `max_iter` — increase if it doesn't converge (try 1000)

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
```

---

### K-Nearest Neighbors (KNN)

**What**: Classifies a point by majority vote of its K nearest neighbors in feature space.

**When to use**:
- Small to medium datasets
- No training needed (lazy learner)
- Odd-shaped decision boundaries

**Advantages**: Simple, no training, naturally multi-class, non-parametric
**Disadvantages**: Slow prediction on large datasets, sensitive to scale (MUST scale), curse of dimensionality

**Key hyperparameters**:
- `n_neighbors` — K (tune with CV; odd K avoids ties)
- `weights` — 'uniform' or 'distance' (closer neighbors vote more)
- `metric` — 'euclidean', 'manhattan', etc.

```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5, weights='uniform')
```

---

### Support Vector Machine (SVM)

**What**: Finds the maximum-margin hyperplane that separates classes. Kernel trick handles non-linearity.

**When to use**:
- Small to medium datasets with many features
- Binary classification (multi-class needs one-vs-rest)
- High-dimensional data (text, images)

**Advantages**: Effective in high dimensions, robust to outliers (uses support vectors only), powerful with kernels
**Disadvantages**: Slow on large datasets, requires scaling, hard to interpret, kernel choice matters

**Key hyperparameters**:
- `C` — regularization (lower=wider margin/more misclassification allowed)
- `kernel` — 'rbf' (default), 'linear', 'poly', 'sigmoid'
- `gamma` — kernel coefficient ('scale' or 'auto' or float) — controls influence radius

```python
from sklearn.svm import SVC
model = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)
```

---

### Decision Tree, Random Forest, Gradient Boosting — Classification

Same algorithms as regression, different output (class probabilities instead of continuous values).

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

dt  = DecisionTreeClassifier(max_depth=5, class_weight='balanced')
rf  = RandomForestClassifier(n_estimators=100, class_weight='balanced')
gb  = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
```

`class_weight='balanced'` — critical for imbalanced datasets. Automatically weights classes inversely proportional to their frequency.

---

## Algorithm Selection Quick Guide

| Problem | Small data | Medium data | Large data |
|---|---|---|---|
| Regression | LinearRegression, Ridge | RandomForest | GradientBoosting |
| Classification | LogisticRegression | RandomForest | GradientBoosting |
| Non-linear | KNN, Decision Tree | RandomForest | GradientBoosting |
| High-dimensional | LogisticRegression, SVM | SVM | LogisticRegression |
| Interpretability needed | LinearRegression, Logistic, DT | — | — |

**Default starting point**: Random Forest for both regression and classification.
It rarely fails completely and gives good baselines with minimal tuning.
