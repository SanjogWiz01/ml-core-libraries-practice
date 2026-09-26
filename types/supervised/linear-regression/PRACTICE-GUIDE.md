# Linear Regression - Practice Guide

Recipes, patterns and traps for using linear regression in real work. The maths
lives in [`THEORY-GUIDE.md`](THEORY-GUIDE.md); this file is about *doing* it.

---

## 1. The 10-Line Baseline You Always Write First

Never skip this. A trained linear model is your yardstick for everything else.

```python
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# The "do nothing" reference: predict the training mean for everything.
dummy = DummyRegressor(strategy="mean")
print("Dummy  R2 :", round(cross_val_score(dummy, X_train, y_train, cv=5).mean(), 4))

# The real baseline.
model = LinearRegression()
cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
print("Linear  R2:", round(cv.mean(), 4), "+/-", round(cv.std(), 4))

model.fit(X_train, y_train)
pred = model.predict(X_test)
print("MAE :", round(mean_absolute_error(y_test, pred), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 3))
print("R2  :", round(r2_score(y_test, pred), 4))
```

If `LinearRegression` cannot beat `DummyRegressor`, the dataset has no linear
signal and you should stop looking at linear models.

---

## 2. Dataset Setup (in this order)

```python
df = pd.read_csv("data.csv")

df.info()                    # dtypes, null counts, memory
df.describe()                # scale, spread, obvious outliers
df.isna().sum()              # missing per column
df["target"].value_counts()  # is the target skewed? skewed target -> log1p
df.corr(numeric_only=True)["target"].sort_values()
```

Cheap wins to try, in rough order of payoff:

| Transform | When |
|-----------|------|
| `log1p(target)` | right-skewed target (prices, income, counts) |
| `StandardScaler` | anything using gradients or penalties; multi-feature models |
| `OneHotEncoder` | categorical features |
| `SimpleImputer(median)` | missing numerics |
| interaction terms | suspected non-additive effect |
| date parts (`year`, `month`, `dayofweek`) | any date column |

---

## 3. Leakage - The Mistake That Destroys Results

**Leakage is any information that would not exist at prediction time flowing
into training.** It produces beautiful, worthless, disappointing-on-arrival
numbers.

Real leaks:

- fitting `StandardScaler` on the **whole** dataset before splitting
- `pd.get_dummies` on the full frame (categories appear that only occur in test)
- target-derived columns: `price_per_sqft` when predicting `price`
- `df.fillna(df['col'].mean())` over train+test together
- dropping rows using information that only appears after the outcome
- target encoding done before the split

The rule: **split first, fit transforms on train only.**

```python
# WRONG - the scaler has seen the test distribution
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# RIGHT - Pipeline guarantees the ordering
pipe = make_pipeline(StandardScaler(), Ridge())
pipe.fit(X_train, y_train)          # scaler fitted on train only
pipe.predict(X_test)                 # test only ever transformed
```

`Pipeline` is the fix, not discipline, because it makes the correct thing the
easy thing.

---

## 4. Feature Engineering Recipes

```python
# Polynomial expansion (still linear regression!)
from sklearn.preprocessing import PolynomialFeatures
PolynomialFeatures(degree=2, include_bias=False)

# Interactions only, no squares
from sklearn.preprocessing import InteractionTerms

# Log target, invert at the end
y_log = np.log1p(y)
model.fit(X_train, y_log)
pred = np.expm1(model.predict(X_test))     # back to original units
# bias correction for retransformation, explained in script 05 notes

# Binning a continuous feature into buckets
pd.cut(df["age"], bins=[0,25,35,45,60,100], labels=False)

# Cyclical time features (hour 23 and 0 are neighbours, not far apart)
df["h_sin"] = np.sin(2*np.pi*df["hour"]/24)
df["h_cos"] = np.cos(2*np.pi*df["hour"]/24)

# Target encoding, done safely: fit on train, map to test
means = train.groupby("city")["price"].mean()
train["city_te"] = train["city"].map(means)
test["city_te"]  = test["city"].map(means).fillna(train["price"].mean())
```

---

## 5. Choosing Regularisation Strength

```python
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True, random_state=42)

alphas = np.logspace(-4, 4, 50)          # 0.0001 -> 10000, log-spaced

ridge = RidgeCV(alphas=alphas, cv=cv).fit(X_train_s, y_train)
print("best alpha:", ridge.alpha_)

lasso = LassoCV(cv=cv, max_iter=10000, random_state=42).fit(X_train_s, y_train)

enet  = ElasticNetCV(l1_ratio=[.1, .5, .9, .95, .99, 1], alphas=alphas, cv=cv, random_state=42)
```

Read the results like this:

- **Best alpha tiny** -> the unregularised model was fine; don't add complexity.
- **Best alpha huge** -> severe overfitting or badly scaled features.
- **Lasso zeroes many coefficients** -> many features are noise, or collinear.
- **A narrow CV band** -> low variance in your estimate, you can trust it.
- **A wide CV band** -> dataset too small, or too much variance; get more data.

---

## 6. Diagnostic Playbook

Run these in order after the first fit.

| Check | Symptom | Usual cause | Fix |
|-------|---------|-------------|-----|
| Residual vs fitted shows a curve | systematic bias | non-linearity | polynomial terms, transform, splines |
| Residuals fan out (funnel) | non-constant variance | heteroscedasticity | `log1p(y)`, weighted fit, robust regression |
| Residuals randomly scattered, R2 low | model too simple | missing signal | add features, interactions, non-linear model |
| Large isolated residuals | outliers | data entry / rare cases | inspect rows, Huber/RANSAC, robust scaler |
| R2 great, MAE terrible | a few huge misses | heavy tail in `y` | MAE-based models, quantile regression |
| Train R2 high, test R2 negative | overfit | too many features | regularise, select features, more data |
| Coefficients flip sign between fits | collinearity | multicollinearity | drop/merge, PCA, ridge |
| VIF > 10 | same problem, confirmed | - | remove the offender |
| Residuals autocorrelated in time | independence violated | data leakage in time | difference features, time-series split |

---

## 7. Deployment Checklist

```python
# 1. Persist preprocessing + model together as ONE artifact
import joblib
joblib.dump(pipe, "model.joblib")

# 2. Reload and smoke-test in a fresh process
loaded = joblib.load("model.joblib")
assert loaded.predict(X_test.head(1))[0] == pytest.approx(pred[0])

# 3. Never let input schema drift
assert list(new_df.columns) == list(train_columns)

# 4. Watch the inputs you trained on, not just the predictions
trained_min, trained_max = X_train[col].min(), X_train[col].max()
if not trained_min <= new_df[col].min() <= trained_max:
    log.warning("out-of-range value for %s: model never saw this", col)

# 5. Log residuals in production; drift shows up there first
```

---

## 8. When to Choose Linear Regression Anyway

- The relationship is genuinely close to linear (`R2` ~0.9 with a small model)
- You must **explain** each prediction to a human (medical, credit, legal)
- You need a **stable** model whose coefficients will not swing between retrains
- Features are many and correlated - linear regression is far less variance-prone
  than a random forest in that regime
- You are a baseline that everything else must beat
- Latency budget is microseconds

When to *not*:

- Clear curvature in the residuals (go polynomial, splines, GBM, a NN)
- Interaction effects dominate
- You need nonlinear boundaries in a classifier (logistic regression is linear
  in the same way; use trees or kernels)
- Outliers are common and unremovable (use a robust estimator)

---

## 9. Interview Questions With Short Answers

**Why is MSE used instead of MAE?**
Smooth and differentiable, strongly penalises large errors, and strictly convex so
there is a unique optimum.

**Why is a low R2 not always bad?**
In some domains almost all variance is noise. What matters is R2 *relative to the
baseline and to the cost of an error*, not the absolute number.

**How do you know the model is overfitting?**
Train error much better than validation error, and the gap grows as you add
features. Cross-validation score deteriorating while training score improves.

**Why does L1 regularisation perform feature selection but L2 does not?**
The L1 constraint region is a diamond with corners on the axes, so the optimum
lands exactly on an axis (weight = 0) for the sparsest solution inside the region.
L2's circle touches no coordinate axis, so weights only shrink toward zero, never
reach it.

**Why does adding features never decrease training R2?**
One more parameter can always do at least as well on the training set, and
`R2_adj` / CV / validation error are what penalise the extra freedom.

**Interpretation of `y = 40 + 2.5*area - 3*age` where area is m2 and age is years.**
Holding age fixed, each extra square metre adds ~2.5 units. Each extra year
*subtracts* 3 units. 40 is the prediction for a 0 m2, 0-year-old house - a
mathematical anchor, rarely meaningful.

**How do you handle a feature in metres and one in kilometres?**
Standardise both, or convert to the same unit. Otherwise the coefficient on the
larger-unit feature is systematically smaller and the L2 penalty acts on it
unequally.

**When is the normal equation preferred over gradient descent?**
Small `n` and `p`, no regularisation needed, and you need the exact solution.
Gradient descent (or the SVD-based `lstsq`) is mandatory once `p` is large or you
need L1/elastic-net penalties.

---

## 10. Common Mistakes Cheat Sheet

1. Fitting a scaler before the split -> leakage
2. Using `r2_score` on the training set to claim performance -> always test/CV
3. `sqrt(mean_squared_error(...))` confusion between MSE and RMSE -> keep labels straight
4. MAPE on targets containing zeros -> infinite error
5. Forgetting `random_state` in `train_test_split` -> irreproducible results
6. Not removing constant columns -> singular Gram matrix
7. Interpreting a coefficient from an unscaled multi-feature model -> meaningless
8. Reporting R2 from a single 80/20 split -> one sample of a high-variance estimate
9. Dropping outliers just because they hurt R2 -> cherry-picking, be transparent
10. Polishing residuals to look random -> the answer is in the residuals, not the plot
