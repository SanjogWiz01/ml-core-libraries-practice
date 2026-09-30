# Logistic Regression

The complete logistic regression track: theory, hand-written implementations, and
runnable scikit-learn code.

Logistic regression is the **first classifier you should always fit**. It is fast,
it has almost no tunable knobs, it produces coefficients you can explain to a
stakeholder, and it hands you *calibrated probabilities* - which almost no other
model gives you for free.

## Why Logistic Regression Still Matters

- It is the reference against which everything else is measured. If a random
  forest gets AUC = 0.81 and logistic regression gets 0.84, the forest is not
  needed.
- It is the only common model where the coefficient has a clean human meaning:
  `exp(w)` is a literal odds ratio. That interpretability is exactly what
  regulated and high-stakes settings require.
- Its probabilities are **calibrated**, which means "0.20" really does mean 20%.
  Random forests, boosted trees and neural nets are all noticeably over-confident
  out of the box.
- It is a **convex** optimisation problem with a unique optimum. It cannot get
  stuck, it cannot overfit through optimisation noise, and it trains on millions
  of rows in seconds.

## The Model

```
z = w0 + w1*x1 + ... + wn*xn          <- the logit, any real number
p = 1 / (1 + exp(-z))                <- P(y = 1 | x), always in (0, 1)
y_hat = 1 if p >= 0.5 else 0         <- thresholded prediction
```

Trained by minimising the **log loss**:

```
J(w) = -(1/n) * SUM_i [ y_i*log(p_i) + (1 - y_i)*log(1 - p_i) ]
```

The gradient is `(1/n) * X^T (p - y)` - the same expression as linear regression,
with the residual swapped for `p - y`.

## Folder Contents

### Guides (read these first)

| File | What it covers |
|------|----------------|
| [`THEORY-GUIDE.md`](THEORY-GUIDE.md) | Maths: sigmoid, log loss, the gradient derivation, `C` and separability, coefficient interpretation, boundary geometry, softmax, all metrics explained |
| [`PRACTICE-GUIDE.md`](PRACTICE-GUIDE.md) | Practical recipes: baseline, leakage, the three outputs, threshold selection from cost, imbalance fixes, tuning `C`, calibration, interview questions, common mistakes |

### Code (9 runnable scripts, numbered in learning order)

| File | Topic |
|------|-------|
| [`01_logistic_regression_from_scratch.py`](01_logistic_regression_from_scratch.py) | Fit by hand with gradient descent, verify against scikit-learn |
| [`02_logistic_regression_sklearn.py`](02_logistic_regression_sklearn.py) | Constructor args, `decision_function` vs `predict_proba` vs `predict`, odds ratios |
| [`03_gradient_descent_variants.py`](03_gradient_descent_variants.py) | Batch vs stochastic vs mini-batch vs Adam, loss vs wall-clock, why scaling matters |
| [`04_decision_boundary_and_geometry.py`](04_decision_boundary_and_geometry.py) | Deriving the boundary, geometric vs functional margin, XOR and circles |
| [`05_loss_functions_and_metrics.py`](05_loss_functions_and_metrics.py) | Log loss, confusion matrix, precision/recall/F1, ROC and PR curves from scratch |
| [`06_probability_thresholds_and_calibration.py`](06_probability_thresholds_and_calibration.py) | Ranking vs calibration, reliability diagrams, Platt/isotonic, thresholds from cost |
| [`07_class_imbalance_handling.py`](07_class_imbalance_handling.py) | Why accuracy inverts, ROC vs PR under imbalance, SMOTE and `class_weight` measured |
| [`08_regularization_l1_l2_elasticnet.py`](08_regularization_l1_l2_elasticnet.py) | What the penalty does to coefficients, coefficient paths, `l1_ratio` in sklearn 1.8 |
| [`09_multiclass_logistic_regression.py`](09_multiclass_logistic_regression.py) | Softmax, hand-written gradient check, the K-1 redundancy trap |

## Setup

```bash
pip install numpy pandas scikit-learn matplotlib
```

Script 07 imports `imblearn` for the SMOTE comparison:

```bash
pip install imbalanced-learn
```

## Run

```bash
cd types/supervised/logistic-regression
python 01_logistic_regression_from_scratch.py
```

Headless machines (CI, servers) should set a non-interactive backend first:

```bash
MPLBACKEND=Agg python 06_probability_thresholds_and_calibration.py   # Linux/macOS
$env:MPLBACKEND="Agg"; python 06_probability_thresholds_and_calibration.py   # Windows PowerShell
```

## Learning Path

```
THEORY-GUIDE.md  (concepts + maths)
        |
        v
01 from scratch  ->  02 sklearn  ->  03 GD variants
        |
        v
04 boundary geometry  ->  05 metrics
        |
        v
06 thresholds + calibration  ->  07 imbalance
        |
        v
08 regularisation  ->  09 multiclass
        |
        v
PRACTICE-GUIDE.md  (the reference book for real work)
```

A sensible order if you are short on time: 01, 05, 07, then
[`PRACTICE-GUIDE.md`](PRACTICE-GUIDE.md). Those four cover the majority of the
decisions you actually have to make.

## Version Notes

Written and verified against **scikit-learn 1.8**, where two familiar arguments
have changed:

- `penalty=` is deprecated on `LogisticRegression`. Choose with `l1_ratio`
  (`0` = L2, `1` = L1, between = elastic net). Script 08 uses this.
- `multi_class=` has been removed. A multiclass fit is always multinomial
  (softmax); use `OneVsRestClassifier` if you specifically want one-vs-rest.
  Script 09 compares them.

## Related

- [`../README.md`](../README.md) - supervised learning overview
- [`../linear-regression/`](linear-regression/) - the regression track; the loss
  gradient is nearly identical and the two read well back to back
