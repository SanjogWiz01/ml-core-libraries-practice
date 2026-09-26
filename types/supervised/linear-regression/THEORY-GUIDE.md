# Linear Regression - Theory Guide

Everything you need to understand linear regression mathematically, written for
someone who wants to *derive* it rather than memorise it.

---

## 1. The Model

For a single feature `x`, linear regression predicts a straight line:

```
y_hat = w0 + w1 * x
```

- `w1` is the **slope**: how much `y` changes when `x` increases by 1.
- `w0` is the **intercept**: the prediction at `x = 0`. This is a modelling
  choice, not necessarily meaningful, but it is forced on you by the model form.

For `p` features we get a **hyperplane** in `p+1` dimensions:

```
y_hat = w0 + w1*x1 + w2*x2 + ... + wp*xp
```

**Superficial but important:** despite the name, "linear" refers to linearity in
the *parameters* `w`, not in the features. `w1*x1*w2` would still be linear
regression - a separable (convex) optimisation problem is what makes it
tractable. This is why `PolynomialFeatures` (see script 11) is still linear
regression, while a random forest is not.

---

## 2. Cost Function - Mean Squared Error

We need a score that is "low is good". The standard choice:

```
MSE = (1/n) * SUM_i (y_i - y_hat_i)^2
```

Why squared error?

- **Penalises big errors much more than small ones.** If you are off by 2, that
  is 4x worse than being off by 1. For a squared physical quantity this matches
  how errors add.
- **Smooth and differentiable everywhere.** No kinks, so gradient descent works
  cleanly (compare MAE, whose gradient is constant-magnitude and therefore
  bouncey).
- **Strictly convex** in `w`, so a single optimum exists. No local minima traps.

The price: **sensitivity to outliers**, because squares amplify them. That single
property is the root of `HuberRegressor`, `RANSAC` and robust scaling
(script 15).

---

## 3. Two Ways to Fit It

### 3a. Closed Form - The Normal Equation

Set the gradient of MSE to zero and solve. In matrix form with a design matrix
`X` (rows are samples, first column all ones for the intercept):

```
w* = (X^T X)^(-1) X^T y
```

- `X^T X` is the **Gram matrix** - a measure of how the features co-vary.
- It is invertible **only if the features are linearly independent**. Duplicate
  or perfectly collinear columns make it singular (script 16).
- We use `pinv` (pseudo-inverse) rather than `inverse` in practice, which
  handles the singular case gracefully via SVD.

**Pros:** exact, one shot, no learning rate, deterministic, fastest for small
`n`/`p`. **Cons:** numerically unstable when `X^T X` is ill-conditioned, and
you must invert a `p x p` matrix - the moment `p` is large (say > 10,000, the
vocabulary size of a text model) it becomes infeasible. It also cannot be
regularised in the way we need. See `04_normal_equation.py`.

### 3b. Iterative - Gradient Descent

Start with random weights, then repeatedly step downhill:

```
prediction  y_hat = X . w
error       e     = y_hat - y
gradient    dW    = (2/n) * X^T . e          # d(MSE)/dw
update      w     = w - learning_rate * dW
```

This is **ordinary least squares (OLS)**, which is exactly what both methods
solve - the iterative version just takes many small steps instead of one exact
jump.

Hyperparameters you now own:
- `learning_rate` - too small = thousands of steps; too large = divergence
  (the loss goes to `inf`/`nan`).
- `n_iterations` - how long to keep stepping.
- `momentum` / `adam` - optional, they add inertia to escape the slow ravine
  that plain GD falls into when features have very different scales (script 03).

`sklearn` exposes all of this through `SGDRegressor`, and the closed form
through `LinearRegression`.

---

## 4. Feature Scaling - Why It Is Not Optional

Consider `x1` in metres (~2) and `x2` in millimetres (~2000). MSE's gradient
along `w2` is roughly 1000x larger than along `w1`, so:

- plain gradient descent zig-zags and converges glacially
- the **regularisation penalty** `lambda * ||w||^2` becomes scale-dependent, so
  effectively penalises `w2` and ignores `w1`
- **coefficients are not comparable** - a larger `|w|` may just mean a larger
  unit, not a stronger effect

Standardise with `StandardScaler` (zero mean, unit variance) or `MinMaxScaler`
(0 to 1) *before* fitting anything with gradients or penalties.

One honest caveat: for a **single-feature OLS prediction**, scaling is
mathematically irrelevant. `y = a*x + b` fitted on `x` gives the same predictions
as fitting on `(x - mean) / std` - only the coefficient's units change
(`w_scaled = w * std(x)`). The slope and intercept always transform this way:

```
w_scaled = w * std(x)
b_scaled = b - w * mean(x)
```

So in script 07, scaling is a *practical* requirement for gradient descent and
for cross-feature coefficient comparison, not a mathematical one for single
variable prediction. That nuance is the difference between memorising and
understanding.

---

## 5. Evaluating a Regression Model

If `y_bar` is the mean of the targets:

```
MAE   = (1/n) SUM |y_i - y_hat_i|
RMSE  = sqrt((1/n) SUM (y_i - y_hat_i)^2)
MAPE  = (100/n) SUM |y_i - y_hat_i| / |y_i|
R2    = 1 - SUM (y_i - y_hat_i)^2 / SUM (y_i - y_bar)^2
```

### R2 in one sentence

`R2` is the fraction of the target's variance the model explains. `R2 = 1` is
perfect, `R2 = 0` is no better than always predicting the mean, and `R2 < 0` is
**worse** than predicting the mean (which happens constantly on unseen data and
tells you the model is not worth deploying).

`R2` is scale free, which is why it is the default metric. `sklearn`'s
`mean_squared_error(..., squared=False)` was removed in 1.6 - use
`root_mean_squared_error` or `np.sqrt(mean_squared_error(...))`.

### Adjusted R2

`R2` always grows when you add features, even useless ones. Adjusted R2
penalises that:

```
R2_adj = 1 - (1 - R2) * (n - 1) / (n - p - 1)
```

It only increases when the new feature brings real signal, so it is a fairer
comparison across different feature counts.

### Which metric when

| Situation | Use |
|-----------|-----|
| General reporting | `R2` + `RMSE` |
| Outliers present, or errors must be comparable in original units | `MAE` |
| You care about *percentage* error (revenue forecasting) | `MAPE` |
| MAPE with `y` near zero | never - it explodes. Use `sMAPE` or `MAE` |

**MAE vs RMSE:** if you would rather treat a 10-unit miss as ten 1-unit misses
(MAE) than as one catastrophic event (RMSE), use MAE. See script 05 for the
numerical demonstration of how differently they react to a single bad point.

---

## 6. The Assumptions of Linear Regression

OLS is **unbiased only if these hold**. Violating them does not break the code -
it breaks your *interpretation*.

1. **Linearity** - `E[y | X]` is linear in `X`. Check: residuals vs fitted
   should show no pattern.
2. **Independence** - one observation does not leak into another. Check: ordered
   residual scatter / autocorrelation. Time series usually violate this.
3. **Constant variance (homoscedasticity)** - residual spread is uniform across
   `X`. Check: residual fan shape. Fix: transform `y` with `log1p`/`sqrt`, or
   model it as regression with weights.
4. **No perfect multicollinearity** - no feature is an exact combination of
   others. The model still fits; it just cannot tell which coefficient is the
   real one. Check: VIF (script 16).
5. **Normally distributed residuals** - needed for *inference* (p-values,
   confidence intervals), not for the predictions themselves. Check: QQ plot.
6. **No influential outliers** - OLS is a high-breakdown-sensitive estimator.
   Check: residual/studentised-residual plot. Fix: Huber, RANSAC (script 15).

Note the asymmetry: assumptions 1-4 mainly threaten *explanation*; 5-6 threaten
*uncertainty estimates*. The point predictions of OLS are much more robust than
its confidence intervals, which surprises most people.

---

## 7. Regularisation - The Bias/Variance Knob

Adding a penalty to MSE:

```
Objective = MSE + lambda * ||w||^2        -> Ridge  (L2)
Objective = MSE + lambda * ||w||_1        -> Lasso (L1)
Objective = MSE + lambda*(a*L1 + (1-a)*L2) -> ElasticNet
```

| | Ridge | Lasso |
|---|-------|-------|
| Penalty | `lambda * SUM wj^2` | `lambda * SUM |wj|` |
| Effect | shrinks all coefficients smoothly | drives some coefficients to **exactly zero** |
| Geometry | circular constraint region | diamond constraint region (corners on axes) |
| Multi-collinear features | splits weight evenly between them | picks one, zeroes the other |
| Feature selection | no | yes, built in |

`lambda = 0` recovers OLS. Large `lambda` underfits. The `coefficient path` -
plotting every coefficient as a function of `lambda` - is the most useful
diagnostic in all of regularisation, and script 12 draws it.

---

## 8. Bias-Variance

```
Expected Test Error = Bias^2 + Variance + Irreducible Noise
```

- **Bias** - the gap between the average prediction over many datasets and the
  truth. Too little model complexity = high bias (underfitting).
- **Variance** - how much predictions move when you retrain on a different
  sample of the same size. Too much complexity = high variance (overfitting).

Linear regression has **low variance, potentially high bias**. It cannot bend,
only tilt, so curved truth is a bias problem. Adding features or polynomial terms
buys flexibility and pays in variance. `degree` in script 06 walks this exact
tradeoff so you can see it.

---

## 9. Solvability Checklist

Before you ship a linear model, be able to answer yes to:

- [ ] Did I split *before* fitting preprocessing?
- [ ] Are residuals vs fitted patternless?
- [ ] Is residual spread roughly uniform (not a fan)?
- [ ] Are all VIFs below ~5-10?
- [ ] Have I checked the QQ plot for extreme tails?
- [ ] Is my test score an average over folds, not one lucky split?
- [ ] Have I compared against a `DummyRegressor` predicting the mean?
- [ ] Do I care about explanation, prediction, or both? (changes everything)

## Formulas Reference Card

| Concept | Formula |
|---------|---------|
| Prediction | `y_hat = X @ w + b` |
| MSE | `(1/n) * ||y - Xw - b||^2` |
| Gradient (with intercept column in X) | `(2/n) * X^T (Xw - y)` |
| Normal equation | `w = (X^T X)^-1 X^T y` |
| Ridge objective | `MSE + lambda * ||w||_2^2` |
| Lasso objective | `MSE + lambda * ||w||_1` |
| R2 | `1 - SS_res / SS_tot` |
| Adjusted R2 | `1 - (1 - R2)(n-1)/(n-p-1)` |
| MAE | `(1/n) * ||y - y_hat||_1` |
| RMSE | `sqrt(MSE)` |
| Coefficient relation under scaling | `w_scaled = w * std(x)`, `b_scaled = b - w*mean(x)` |
