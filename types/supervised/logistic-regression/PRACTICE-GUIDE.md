# Logistic Regression - Practice Guide

Recipes, patterns and traps for using logistic regression in real work. The maths
lives in [`THEORY-GUIDE.md`](THEORY-GUIDE.md); this file is about *doing* it.

---

## 1. The 12-Line Baseline You Always Write First

Never skip this. A fitted classifier is your yardstick for everything else.

```python
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# The "do nothing" reference: always predict the majority class.
dummy = DummyClassifier(strategy="prior")
print("Dummy  acc:", round(cross_val_score(dummy, X_train, y_train, cv=5,
                                           scoring="accuracy").mean(), 4))

# The real baseline.
model = LogisticRegression(max_iter=1000)
cv = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
print("LogReg  AUC:", round(cv.mean(), 4), "+/-", round(cv.std(), 4))

model.fit(X_train, y_train)
pred = model.predict(X_test)
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))
```

Two details that are not optional:

- **`stratify=y`** on the split. Without it a small class can end up unevenly
  distributed, and your test numbers become unrepeatable noise.
- **Compare against `DummyClassifier`**, not against 50%. On any problem where
  the classes are lopsided, "beat the majority-class baseline" is the bar.

---

## 2. Dataset Setup (in this order)

```python
df = pd.read_csv("data.csv")

df.info()                     # dtypes, null counts, memory
df.isna().sum()               # missing per column
df["target"].value_counts(normalize=True)   # the single most important line
df["target"].value_counts()                 # absolute counts too
df.describe()                 # scale, spread, obvious outliers
df.corr(numeric_only=True)["target"].sort_values()
```

Read `value_counts()` before anything else. It tells you whether accuracy is even
a legal metric, and it is the number that decides your whole evaluation strategy.

| Transform | When |
|-----------|------|
| `StandardScaler` | anything penalised, iterative, or multi-feature |
| `OneHotEncoder(handle_unknown="ignore")` | categorical features |
| `SimpleImputer(strategy="median", add_indicator=True)` | missing numerics |
| `log1p` on features with a heavy tail | income, counts, transaction size |
| date parts (`year`, `month`, `dayofweek`) | any date column |

---

## 3. Leakage - The Mistroke That Destroys Results

**Leakage is any information that would not exist at prediction time flowing
into training.** It produces beautiful, worthless, disappointing-on-arrival
numbers.

Real leaks in classification:

- fitting `StandardScaler` on the **whole** dataset before splitting
- `pd.get_dummies` on the full frame (categories appear that only occur in test)
- target-derived columns: `chargeback_rate` when predicting `chargeback`
- target encoding done before the split
- `df.fillna(df['col'].mean())` over train+test together
- dropping rows using information that only appears after the outcome

The rule: **split first, fit transforms on train only.**

```python
# WRONG - the scaler has seen the test distribution
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# RIGHT - Pipeline guarantees the ordering
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
pipe.fit(X_train, y_train)          # scaler fitted on train only
pipe.predict(X_test)                 # test only ever transformed
```

`Pipeline` is the fix, not discipline, because it makes the correct thing the
easy thing.

---

## 4. Three Outputs, Three Different Things

The most common misunderstanding in classification is treating these as one thing:

```python
z = model.decision_function(X_test)   # the LOG-ODDS, unbounded, signed
p = model.predict_proba(X_test)       # PROBABILITIES, sum to 1 across classes
y_hat = model.predict(X_test)         # LABELS, p >= 0.5 after argmax
```

| Output | What it is | Use it for |
|--------|-----------|------------|
| `decision_function` | signed distance to the boundary, in log-odds | margins, custom thresholds, stacking |
| `predict_proba[:, 1]` | P(y=1) | ranking, calibration, expected cost |
| `predict` | a hard label | reporting, actions with no notion of degree |

Mixing them up is how people end up thresholding a probability at 0.5 when they
meant a log-odds of 0.5, or reporting a label count as if it were a confidence.

---

## 5. Choosing the Threshold

`predict` hard-codes 0.5. That is only correct when the error costs are equal and
the base rate is 50%. Usually neither holds.

```python
# Predict positive only when the expected cost is favourable.
p = model.predict_proba(X_test)[:, 1]
pred = (p > 0.08).astype(int)     # favour recall when missing a case is costly

# Build the trade-off curve yourself
from sklearn.metrics import precision_recall_curve
prec, rec, thr = precision_recall_curve(y_test, p)
```

Pick the threshold from the **cost matrix**, not from whichever metric flatters
the model:

```
expected_cost(t) = FN_cost * FN_rate(t) + FP_cost * FP_rate(t)
```

In a medical screen, `FN_cost` is usually 10-100x `FP_cost`, so the threshold
belongs well below 0.5. In spam filtering it is the reverse. `0.5` is a default,
not a decision.

**Sanity check:** after choosing, compare your positive-prediction rate against
the operational flag rate you can actually staff. If your model flags 30% of
transactions and your team reviews 2%, you have a problem the accuracy number
never showed you (script 06 works the arithmetic).

---

## 6. Imbalance - What Actually Helps

Real base rates run from 0.1% (fraud) to 10% (churn). Everything interesting
happens in that range.

**First: change the metric, not the model.**

| Base rate | Accuracy | Use instead |
|-----------|----------|-------------|
| ~50% | fine | accuracy + confusion matrix |
| 10-50% | fine | accuracy + F1 |
| 1-10% | misleading | PR-AUC, recall, confusion matrix |
| <1% | actively harmful | PR-AUC, recall at a fixed flag rate |

**Then, if you still need to:** resampling barely changes the *ranking* and
noticeably breaks the *calibration*.

```python
from sklearn.utils.class_weight import compute_sample_weight

# Best default: tell the loss function to care more about the minority class.
w = compute_sample_weight(class_weight="balanced", y=y_train)
model.fit(X_train, y_train, sample_weight=w)
# or, equivalently inside the estimator:
model = LogisticRegression(class_weight="balanced")

# SMOTE - synthesises minority rows. Only works on dense, numeric features.
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
```

`class_weight="balanced"` is **minority replication in disguise** - it weights
each minority row by `n/(2*n_minority)`. That is why it is preferable to
undersampling: it throws no data away, and it keeps the natural variance of the
majority instead of shrinking the sample to the minority count (script 07
proves the equivalence).

**The catch:** all three approaches bias the predicted probabilities upward,
because the model now optimises a reweighted objective. If you report
probabilities, correct for the prior:

```
p_true = p_adj / (p_adj + (1 - p_adj) * (pi_adj / pi_true))
```

where `pi` is the positive rate. Script 07 derives it. Reporting raw
reweighted probabilities is one of the most common ways to ship a model whose
probabilities are quietly wrong.

---

## 7. Tuning C and the Penalty

`C` is inverse regularisation strength. It is the one hyperparameter that
usually matters.

```python
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

logit = LogisticRegressionCV(
    Cs=np.logspace(-4, 4, 25),       # 0.0001 -> 10000
    cv=cv,
    scoring="neg_log_loss",          # tune on the LOSS, not on accuracy
    max_iter=2000,
)
logit.fit(X_train, y_train)
print("best C:", logit.C_)
```

Use **`StratifiedKFold`**, never plain `KFold` - under imbalance an unstratified
fold can contain almost no positives, making the score meaningless.

Tune on `neg_log_loss` rather than accuracy. Accuracy is a step function of the
threshold, so it barely moves when you change `C`; log loss is what the model
actually minimises, and it is threshold-independent.

Read the result:

- **Best `C` tiny** -> heavily overfit; the problem had little signal, or you need more features
- **Best `C` huge** -> under-regularised, or the classes are separable
- **Best `C` at the grid edge** -> widen the grid before believing it

**sklearn 1.8:** `penalty=` is deprecated; choose with `l1_ratio`
(`0` = L2, `1` = L1). `LogisticRegressionCV(..., Cs=...)` with `l1_ratio=0` is
the L2 path.

---

## 8. Reading the Coefficients

```python
# Coefficient -> log-odds -> odds ratio
for name, w in zip(feature_names, model.coef_[0]):
    print(f"{name:>20}  w={w:+.3f}  odds_ratio={np.exp(w):.2f}")

# The prediction shift, which is the number stakeholders actually want
m, s = X_train[col].mean(), X_train[col].std()
delta = w * s                                  # one standard deviation
print(f"one SD of {col}: log-odds {delta:+.3f}, "
      f"odds x{np.exp(delta):.2f}, "
      f"p {logistic(base_logit + delta):.3f} vs {logistic(base_logit):.3f}")
```

Report the probability shift, not the log-odds. `exp(w)` doubling the odds is
impressive-sounding and meaningless on its own - doubling odds from 1:100 to
1:50 changes `p` from 0.0099 to 0.0198, which nobody cares about. Always anchor
to a realistic `base_logit` before quoting a probability.

---

## 9. Diagnostic Playbook

| Check | Symptom | Usual cause | Fix |
|-------|---------|-------------|-----|
| Train AUC near 1, test AUC much lower | overfit | too many features, or `C` too large | lower `C`, select features, more data |
| AUC near 0.5 everywhere | no signal | weak features, or label noise | new features, check the label definition |
| Accuracy high, recall ~0 | defaulting to majority | imbalance + 0.5 threshold | lower threshold, PR-AUC, `class_weight` |
| Perfect training probabilities | separable data | no regularisation | raise regularisation (lower `C`) |
| Probabilities all near 0.5 | underfit | model too weak, or `C` too small | add features, raise `C`, non-linear model |
| Coefficients flip sign between fits | collinearity | correlated features | drop/merge, or use L2 to stabilise |
| Good AUC, terrible calibration | over-confident | small `C` penalty not enough | calibrate: `CalibratedClassifierCV` |
| Test set has ~0 positives | unlucky split | no `stratify` | `train_test_split(..., stratify=y)` |
| Class weights help recall, wreck calibration | reweighting bias | expected | correct for the prior |

---

## 10. Calibration - Only If You Ship Probabilities

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

# Global: re-regularise instead. Cheapest fix, rarely enough.
# Local: Platt (sigmoid) or isotonic, fitted on held-out data.
cal = CalibratedClassifierCV(model, method="isotonic", cv=5)
cal.fit(X_train, y_train)

print("Brier before:", round(brier_score_loss(y_test, p_raw), 4))
print("Brier after :", round(brier_score_loss(y_test, cal.predict_proba(X_test)[:, 1]), 4))
```

Isotonic needs a lot of data - it fits a free-form monotone curve and will
overfit badly below a few thousand rows. Platt (2 parameters) is the safe
default for small data.

**The leak:** fitting the calibrator on the same rows you evaluate on. Use
`cv=` inside `CalibratedClassifierCV` (as above) or an explicit calibration
split. Otherwise you are measuring how well you memorised, not how well you
calibrate.

---

## 11. Multiclass

```python
# sklearn 1.8: always multinomial (softmax). `multi_class=` was removed.
model = LogisticRegression(max_iter=1000)

# One-vs-rest, only if you need per-class calibration or a sparse K
from sklearn.multiclass import OneVsRestClassifier
ovr = OneVsRestClassifier(LogisticRegression(max_iter=1000))
```

Read the confusion matrix per class, never pooled - a pooled number hides which
class is failing. `model.classes_` gives you the label order that
`predict_proba` columns follow, and getting that mapping backwards is a
classic silent bug.

To interpret coefficients, pick one class as the reference and report the other
`K-1` against it. Interpreting all `K` vectors independently double-counts,
because the multinomial parameterisation is over-determined by exactly one
degree of freedom.

---

## 12. Deployment Checklist

```python
import joblib

# 1. Persist preprocessing + model together as ONE artifact
joblib.dump(pipe, "model.joblib")

# 2. Reload and smoke-test in a fresh process
loaded = joblib.load("model.joblib")
assert loaded.predict(X_test.head(1))[0] == pred[0]

# 3. Never let input schema drift
assert list(new_df.columns) == list(train_columns)

# 4. Monitor calibration drift, not just accuracy
#    If the input mix shifts, the probabilities go stale first.
```

For a model that will be scored in production, also record:

- the training base rate, so you can correct probabilities when the live base rate differs
- the intended threshold and the expected positive rate at that threshold
- the `roc_auc` and `pr_auc` on a held-out set, so drift is detectable

---

## 13. When to Choose Logistic Regression Anyway

- You must **explain** each prediction (medical, credit, legal, fraud review)
- You need **calibrated probabilities**, not just a ranking
- Features are many and correlated - logistic regression is far less
  variance-prone than a random forest in that regime
- You want a **strong baseline** that everything else must beat
- Training data is large (`n` in the millions) and latency budget is tight
- You need a model that will still be in production in three years

When to *not*:

- The true boundary is non-linear (XOR, concentric rings) - use trees, kernels,
  or add polynomial features
- Interactions dominate
- You have unstructured data (text, images) - linear models on those are
  usually a baseline step, not a destination

---

## 14. Interview Questions With Short Answers

**Why can't you use linear regression for classification?**
OLS on 0/1 targets predicts outside [0,1], MSE is not a proper scoring rule for
Bernoulli outcomes, and it fails to penalise confident mistakes properly.

**What does the coefficient actually mean?**
The change in **log-odds** per one unit of the feature. `exp(w)` is the odds
ratio, which is the human-readable form.

**Why is `predict()` just thresholding `predict_proba()` at 0.5?**
`predict` takes `argmax` of the probabilities, which equals thresholding the
positive probability at 0.5 in binary classification. It is a hard-coded default,
not a decision - choose your own threshold from the cost matrix.

**When is ROC-AUC misleading?**
Under severe class imbalance. Its baseline is 0.5 regardless of base rate, so a
model that catches almost no positives can still score well. Use PR-AUC, whose
baseline is the positive rate.

**Why must you scale features before L1/L2 logistic regression?**
The penalty `lambda*||w||^2` is scale-dependent, so without scaling it
effectively penalises whichever feature is measured in the largest units.

**What causes the log loss to diverge to infinity?**
Complete separability. The coefficients grow without limit because the loss has
no minimum. Fix with regularisation (lower `C`) or collect overlapping data.

**What is `class_weight='balanced'` doing?**
Weighting each row inversely to its class frequency, which is mathematically
equivalent to replicating minority rows. It changes the loss, not the data -
and it biases the predicted probabilities upward.

**Why can't logistic regression solve XOR?**
The XOR boundary is not a hyperplane. No choice of weights produces it, because
logistic regression's class boundary is always linear in the features.

**Why does `decision_function` exist when `predict_proba` exists?**
The log-odds are unbounded and preserve the *magnitude* of the decision, which
is what you want for margins, stacking meta-features, and custom thresholds.
Probabilities saturate at 0 and 1 and lose that information.

---

## 15. Common Mistakes Cheat Sheet

1. Reporting accuracy on an imbalanced problem -> inverted signal
2. `train_test_split` without `stratify=y` -> unrepeatable, meaningless test numbers
3. Fitting a scaler before the split -> leakage
4. Forgetting `random_state` -> irreproducible results
5. `KFold` instead of `StratifiedKFold` for classification -> folds with no positives
6. Shipping reweighted probabilities without the prior correction -> quietly wrong
7. Treating `|coefficient|` as importance on unscaled data -> unit artefact
8. Accepting the 0.5 threshold without a cost matrix -> arbitrary
9. Reporting probabilities without a reliability diagram -> unverified claims
10. Forgetting `max_iter` -> `ConvergenceWarning`, silently underfit model
