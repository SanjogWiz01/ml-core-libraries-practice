# Linear Regression

The complete linear regression track: theory, hand-written implementations, and
runnable scikit-learn code.

Linear regression is the **first model you should always fit**. It is fast, it
has almost no tunable knobs, it produces interpretable coefficients, and it acts
as the baseline that every fancier model has to beat.

## Why Linear Regression Still Matters

- It is the reference against which everything else is measured. If a random
  forest gets R2 = 0.80 and linear regression gets 0.82, the forest is not needed.
- Its coefficients are readable: `price = 3.1 * area - 12 * age + 40`. That
  interpretability is exactly what regulated and high-stakes settings require.
- A massive share of real datasets are, in fact, close to linear. If the residual
  plot looks flat, complexity is not the missing ingredient.

## The Model

```
y_hat = w0 + w1*x1 + w2*x2 + ... + wn*xn
```

- `y_hat` - predicted value
- `w0` (intercept) - value predicted when every feature is zero
- `wi` (coefficients / weights) - change in prediction per one unit of `xi`

Trained by minimising the **Mean Squared Error**:

```
MSE = (1/n) * SUM_i (y_i - y_hat_i)^2
```

## Folder Contents

### Guides (read these first)

| File | What it covers |
|------|----------------|
| [`THEORY-GUIDE.md`](THEORY-GUIDE.md) | Maths: least squares, normal equation, gradient descent, closed-form vs iterative, assumptions, all evaluation metrics explained |
| [`PRACTICE-GUIDE.md`](PRACTICE-GUIDE.md) | Practical recipes: loading data, leakage, scaling, polynomial features, regularisation, diagnostics, interview questions, common mistakes |

### Code (17 runnable scripts, numbered in learning order)

| File | Topic |
|------|-------|
| [`01_linear_regression_from_scratch.py`](01_linear_regression_from_scratch.py) | Fit a line with hand-written gradient descent, watch the loss fall |
| [`02_linear_regression_sklearn.py`](02_linear_regression_sklearn.py) | The 5-minute scikit-learn version, coefficient interpretation |
| [`03_gradient_descent_variants.py`](03_gradient_descent_variants.py) | Batch vs stochastic vs mini-batch vs Adam, measured |
| [`04_normal_equation.py`](04_normal_equation.py) | Closed-form solution with numpy, precision and conditioning limits |
| [`05_cost_functions_and_metrics.py`](05_cost_functions_and_metrics.py) | MSE, RMSE, MAE, R2, Adjusted R2, MAPE by hand |
| [`06_bias_variance_diagnosis.py`](06_bias_variance_diagnosis.py) | Underfit/just-right/overfit curve, finding the sweet spot |
| [`07_feature_scaling_importance.py`](07_feature_scaling_importance.py) | Why unscaled features break SGD and coefficient comparison |
| [`08_multivariate_regression.py`](08_multivariate_regression.py) | Many features, coefficient ranking, duplicate-feature trap |
| [`09_data_preparation_pipeline.py`](09_data_preparation_pipeline.py) | Leakage-free `Pipeline` + `ColumnTransformer` |
| [`10_train_test_split_and_cross_validation.py`](10_train_test_split_and_cross_validation.py) | Splits, K-fold CV, learning curves, honest error estimates |
| [`11_polynomial_features.py`](11_polynomial_features.py) | Non-linearity, degree selection, extrapolation trap |
| [`12_regularization_ridge_lasso_elasticnet.py`](12_regularization_ridge_lasso_elasticnet.py) | Ridge/Lasso/ElasticNet, coefficient paths, `ElasticNetCV` |
| [`13_residual_analysis_and_diagnostics.py`](13_residual_analysis_and_diagnostics.py) | Residual plots, QQ plot, heteroscedasticity, autocorrelation |
| [`14_feature_importance_and_selection.py`](14_feature_importance_and_selection.py) | RFE, `SelectKBest`, L1 selection, permutation importance |
| [`15_outliers_and_robust_regression.py`](15_outliers_and_robust_regression.py) | Why outliers wreck OLS, and Huber/RANSAC fixes |
| [`16_multicollinearity_detection.py`](16_multicollinearity_detection.py) | Variance Inflation Factor, coefficient instability |
| [`17_complete_production_pipeline.py`](17_complete_production_pipeline.py) | End-to-end: load -> tune -> evaluate -> save -> reload -> serve |

## Setup

```bash
pip install numpy pandas scikit-learn matplotlib joblib
```

## Run

```bash
cd types/supervised/linear-regression
python 01_linear_regression_from_scratch.py
```

Headless machines (CI, servers) should set a non-interactive backend first:

```bash
MPLBACKEND=Agg python 17_complete_production_pipeline.py   # Linux/macOS
$env:MPLBACKEND="Agg"; python 17_complete_production_pipeline.py   # Windows PowerShell
```

## Learning Path

```
THEORY-GUIDE.md  (concepts + maths)
        |
        v
01 from scratch  ->  02 sklearn  ->  04 normal equation  ->  03 GD variants
        |
        v
05 metrics  ->  06 bias/variance  ->  07 scaling
        |
        v
09 pipeline  ->  10 cross-validation  ->  08 multivariate
        |
        v
11 polynomial  ->  12 regularisation
        |
        v
13 diagnostics  ->  14 feature selection  ->  15 outliers  ->  16 multicollinearity
        |
        v
17 production pipeline   (PRACTICE-GUIDE.md as the reference book)
```

## Related

- [`../README.md`](../README.md) - supervised learning overview
- [`../../../random forest/`](../../../random%20forest/) - a non-linear ensemble to compare against
