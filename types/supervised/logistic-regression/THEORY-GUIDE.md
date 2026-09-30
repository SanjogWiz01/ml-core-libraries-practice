# Logistic Regression - Theory Guide

Everything you need to understand logistic regression mathematically, written for
someone who wants to *derive* it rather than memorise it.

---

## 1. The Model

Linear regression predicts a number. We want a probability in `[0, 1]`, so the
first move is to score linearly and then squash:

```
z_i = w0 + w1*x1_i + ... + wp*xp_i          <- the "logit", any real number
p_i = sigmoid(z_i) = 1 / (1 + exp(-z_i))    <- P(y_i = 1 | x_i), in (0, 1)
```

`z` is unbounded, `p` is not. That asymmetry is the whole design: the model has
full freedom in the direction and steepness of the boundary, and the link
function only has to map reals into valid probabilities.

---

## 2. Why Not Just Regress 0/1 Directly

Three reasons, in increasing order of how much they matter.

**The predictions leave [0,1].** OLS on 0/1 targets happily returns -0.2 and 1.4.
You would have to clip, and clipping destroys gradients exactly where you need
them.

**MSE punishes confident correct answers.** For `y = 1, p = 0.99`, MSE is tiny.
For `y = 0, p = 0.01`, also tiny. Log loss does the opposite - it is almost flat
when you are right, and explodes when you are confidently wrong. That asymmetry
is what drives the model to be *calibrated* rather than merely accurate.

**MSE is not a proper scoring rule for Bernoulli outcomes.** The squared error
does not correspond to a proper likelihood, so the minimiser is not the true
conditional mean of `y | x`. The log loss does.

---

## 3. The Cost Function - Log Loss (Cross-Entropy)

For each row, penalise according to how wrong you are:

```
J(w) = -(1/n) * SUM_i [ y_i*log(p_i) + (1 - y_i)*log(1 - p_i) ]
```

Two cases worth reading separately:

- If `y_i = 1`, the term is `-log(p_i)`. This is a **surprise** penalty: `-log(0.95) = 0.05`, but `-log(0.01) = 4.6`. Confidence is rewarded, error is punished superlinearly.
- If `y_i = 0`, symmetric.

Substituting the sigmoid and simplifying gives the numerically friendlier form
used throughout the scripts:

```
J(w) = (1/n) * SUM_i log(1 + exp(-y_i * z_i))      with y_i in {-1, +1}
```

`log(1 + exp(-t))` never overflows for `t > 0`, which is why the label set
`{-1, +1}` is convenient here.

---

## 4. The Gradient - and Why It Is So Clean

Differentiating and simplifying, the `sigmoid` terms collapse:

```
dJ/dw_j = (1/n) * SUM_i (p_i - y_i) * x_ij        <- vectorised: X^T (p - y) / n
```

This is the **same expression as linear regression**, with one substitution:
the residual `y_hat - y` becomes `p - y`. Everything you learned about gradient
descent for regression transfers directly (script 03 measures this).

The cancellation is not a coincidence. For the log loss, `dp/dz = p*(1-p)`, and
the chain rule's `p*(1-p)` factor exactly cancels the `1/(p*(1-p))` that comes
out of `-log(p)` when `y = 1`. This is why the log loss has a linear-time,
constant-memory gradient and why it became the default loss for neural networks.

---

## 5. Fitting It

### 5a. Gradient Descent

```
prediction  z = X . w + b
probability p = sigmoid(z)
update      w = w - learning_rate * (X^T (p - y) / n)
```

Same algorithm as regression, different loss. **You must scale your features.**
Unscaled columns make the loss surface a long ravine and plain GD zig-zags
(script 03 demonstrates the failure so you recognise it).

### 5b. What scikit-learn Actually Does

`LogisticRegression` does **not** use plain gradient descent. It uses
`solver="lbfgs"` by default - Newton-style, which builds an approximate Hessian
and converges in far fewer steps, with no learning rate to tune.

Practical consequence: **scaling still matters, but for a different reason.** It
no longer affects convergence speed; it affects which solution you land on when
features are correlated, and whether coefficients are comparable.

### 5c. The Intercept Is Not Optional

The model cannot represent a constant base rate without one. Without `b`, every
boundary passes through the origin, so a 99%-positive dataset would be
impossible to fit. `sklearn` fits it by default.

---

## 6. Separability - Why `C` Can Run Away

If the classes are **perfectly linearly separable**, the log loss can be driven
arbitrarily close to zero by growing `|w|`. The coefficients diverge, and the
predicted probabilities saturate to exactly 0 and 1.

This is not a bug, it is the loss having no minimum. scikit-learn's defence is
`C`, the inverse of regularisation strength:

```
minimise   C * SUM_i log_loss_i + 0.5 * ||w||^2
```

- Large `C` (default `1.0`) = weak penalty = nearly unregularised
- Small `C` = strong penalty = small weights, modest probabilities
- `C = np.inf` = the unpenalised fit (see script 08 for the sklearn 1.8 spelling)

On separable data a large `C` gives you a model that is *right* and *useless* -
it will confidently predict 0.999999 for every positive, and be wrong with total
certainty the first time the world shifts. Script 06 works through that failure.

---

## 7. Interpreting Coefficients

The coefficient is on the **log-odds**, not the probability. The chain:

```
w_j                                    per one unit of x_j, in log-odds
exp(w_j)                               the ODDS RATIO - "x_j multiplies the odds by this"
odds = exp(z)                          odds of y=1 vs y=0 at the current x
p = odds / (1 + odds)                  and finally the probability
```

Worked example. Suppose `w_income = 0.7` for a feature measured in thousands:

- one extra thousand in income raises log-odds by 0.7
- multiplies the odds by `exp(0.7) ≈ 2.01` - **doubles the odds**
- at odds of 1:1 (p = 0.5), that moves p to `2/3 ≈ 0.67`

Two traps that follow from this:

**`|w|` is not importance.** A feature with a large coefficient may just be
measured in tiny units. On unscaled data, `w` tells you about units, not
influence. Standardise, or use `|w|` only after scaling (script 02).

**The intercept is usually meaningless.** It is the log-odds at every feature
equals zero. For "months employed = 0" or "income = 0" that is not a real
population, so do not read anything into it.

**Odds ratios only move probabilities meaningfully near the middle.** Going from
p = 0.01 to p = 0.02 is a doubling that looks dramatic in relative terms and
does almost nothing in absolute ones. This is why log-odds models appear to
underperform on rare events (script 06).

---

## 8. Decision Boundary and Geometry

Predicting 1 when `p >= 0.5` means predicting 1 when `z >= 0`:

```
w . x + b = 0
```

The boundary is a **hyperplane**, perpendicular to `w`. Two consequences:

- **Orientation** is controlled by the *direction* of `w`.
- **Confidence** is controlled by the *magnitude* of `w`. Scaling `w` by 10 does not move the boundary at all; it just makes every prediction more extreme. The boundary's *position* depends only on `w/|w|`.

### Geometric vs functional margin

The **geometric margin** is the perpendicular distance from the boundary to the
nearest data point: `1/|w|` (after accounting for `b`). A larger `|w|` is a
wider margin, a more robust boundary, and better-behaved probabilities - all
three come from the same knob.

The **functional margin** is `y_i * z_i`, which mixes orientation and scale, and
is what the SVM maximises directly.

### What cannot be separated

XOR and concentric circles have no linear boundary. Logistic regression cannot
represent them, and adding features does not change that unless the added
features are themselves non-linear - `PolynomialFeatures` is the one move that
stays inside the linear family. Script 04 draws all three cases.

---

## 9. Evaluating a Classifier

From the confusion matrix (`TP`, `FP`, `TN`, `FN`):

```
Accuracy  = (TP + TN) / total
Precision = TP / (TP + FP)          "of what I flagged, how much was right?"
Recall    = TP / (TP + FN)          "of what was truly positive, how much did I catch?"
F1        = 2 * P * R / (P + R)      harmonic mean - punishes imbalance between P and R
```

### The metric that must not be accuracy

On a 6%-positive problem, predicting "not fraud" every time scores **94%**
accuracy and catches nothing. Accuracy is not merely uninformative under
imbalance, it is inverted. Script 07 opens with this.

### ROC-AUC vs PR-AUC

- **ROC-AUC** = P(rank a random positive above a random negative). Threshold-independent. Its no-skill baseline is **0.5**.
- **PR-AUC** = a summary of the precision-recall curve. Its baseline is the **positive rate** itself - so on a 6% problem, a useless model scores about 0.06, not 0.5.

That difference is the entire reason PR curves exist: ROC-AUC flatters a model
on severe imbalance because the huge true-negative count keeps the false
positive rate small no matter how bad the model is. Script 07 measures both on
identical scores.

### Rank interpretation of ROC-AUC

ROC-AUC is the **Mann-Whitney U statistic** - the probability that a randomly
chosen positive outranks a randomly chosen negative. That makes it threshold-free
but also discards all calibration information. Ranking and calibration are
independent properties, fixed with different tools (script 06).

---

## 10. Ranking vs Calibration

The model gives you two outputs, and they fail independently:

```
RANKING     are the high-probability rows actually the likely ones?
            Measured by ROC-AUC / PR-AUC. Threshold-independent.

CALIBRATION when it says 0.20, do 20% of those rows turn out to be 1?
            Measured by the reliability diagram, Brier score, log loss.
```

- A model can rank perfectly and lie about every probability (over-confident).
- A model can be perfectly calibrated and rank terribly (predicts 0.5 for everything).

Repairs are different tools: re-regularisation or Platt/isotonic scaling for
calibration; the features themselves for ranking. Conflating them is the root of
most bad threshold choices.

**Brier score** = mean squared error of the probabilities, a proper scoring rule:
```
Brier = (1/n) * SUM_i (p_i - y_i)^2
```

---

## 11. Multiclass - Softmax

Binary gives one number then squashes it. `K` classes gives `K` numbers,
normalised:

```
z_k = w_k . x + b_k                    k-th logit
p_k = exp(z_k) / SUM_j exp(z_j)        softmax
```

The binary case is a special case: softmax with 2 classes reduces algebraically
to the sigmoid. Nothing else changes.

Two things people get wrong:

- **K sets of coefficients hold K-1 sets of information.** The multinomial
  model is over-parameterised by one degree of freedom; the last class's weights
  are a linear combination of the others. Interpreting all `K` coefficient
  vectors independently double-counts. Use one class as the reference.
- **In sklearn 1.8, `multi_class=` has been removed.** A multiclass fit is
  always multinomial. To get one-vs-rest you must ask for it explicitly with
  `OneVsRestClassifier`. For well-separated, balanced data the two agree; the
  OvR model is still per-class calibrated, which sometimes matters (script 09).

Numerical stability: subtract `max(z)` before exponentiating. `exp(z_k)` overflows
for large `z_k`; the ratio is unchanged and the overflow disappears.

---

## 12. Regularisation

```
L2 (ridge)     C * SUM log_loss + 0.5 * ||w||^2
L1 (lasso)     C * SUM log_loss +     ||w||_1
Elastic net    C * SUM log_loss + l1_ratio*||w||_1 + 0.5*(1-l1_ratio)*||w||^2
```

| | L2 | L1 |
|---|----|----|
| Penalty | `0.5*SUM wj^2` | `SUM |wj|` |
| Effect | shrinks all coefficients smoothly | drives some to **exactly zero** |
| Geometry | circular constraint region | diamond - corners on the axes |
| Collinear features | splits weight evenly | picks one, zeroes the other |
| Feature selection | no | yes, built in |

**sklearn 1.8 note:** `penalty=` is deprecated. Choose with `l1_ratio`
(`0` = L2, `1` = L1, between = elastic net), which is what script 08 uses.

The penalty is scale-dependent: without standardisation, L2 effectively
penalises whoever is measured in the largest units. **Always scale before
fitting a penalised logistic model.**

---

## 13. Bias-Variance

```
Expected Test Error = Bias^2 + Variance + Irreducible Noise
```

Logistic regression has **low variance, potentially high bias** - same argument
as linear regression. It cannot bend its boundary, only tilt it, so curved
truth is a bias problem. `C` (or `alpha`) is the knob that trades them.

The reason it is still the workhorse despite that limitation: the bias is
predictive *and* explainable, the variance is tiny, training is a convex problem
with a unique optimum, and it trains on millions of rows in seconds.

---

## 14. Solvability Checklist

Before you ship a classifier, be able to answer yes to:

- [ ] Did I split *before* fitting preprocessing?
- [ ] Did I beat `DummyClassifier(strategy="prior")`?
- [ ] If classes are imbalanced, did I report PR-AUC, not accuracy?
- [ ] Is there a `confusion_matrix` printed, not just one number?
- [ ] If I report probabilities, have I looked at a reliability diagram?
- [ ] Was the threshold chosen from cost, not from F1?
- [ ] Was `C`/`l1_ratio` cross-validated rather than left at the default?
- [ ] Are the classes separable? (if yes, `C` is fighting you)
- [ ] Are my features scaled?
- [ ] Is this the default branch? (the one you already validated)

## Formulas Reference Card

| Concept | Formula |
|---------|---------|
| Logit | `z = w . x + b` |
| Sigmoid | `p = 1 / (1 + exp(-z))` |
| Log loss | `-(1/n) * SUM [y*log(p) + (1-y)*log(1-p)]` |
| Stable log loss | `(1/n) * SUM log(1 + exp(-y*z))`, `y in {-1,+1}` |
| Gradient | `(1/n) * X^T (p - y)` |
| Boundary | `w . x + b = 0` |
| Geometric margin | `1 / \|\|w\|\|` |
| Odds ratio | `exp(w_j)` |
| Softmax | `p_k = exp(z_k) / SUM_j exp(z_j)` |
| Accuracy | `(TP+TN) / total` |
| Precision | `TP / (TP+FP)` |
| Recall | `TP / (TP+FN)` |
| F1 | `2PR / (P+R)` |
| Brier | `(1/n) * SUM (p_i - y_i)^2` |
| ROC-AUC baseline | `0.5` |
| PR-AUC baseline | positive rate `pi` |
| L2 objective | `C*logloss + 0.5*\|\|w\|\|_2^2` |
| L1 objective | `C*logloss + \|\|w\|\|_1` |
