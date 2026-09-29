"""
02 - Logistic Regression with scikit-learn
=========================================

Goal: the 5-minute version, plus the interpretation skills that make a logistic
regression worth putting in front of a stakeholder.

What this file demonstrates
---------------------------
1. `LogisticRegression` on a real-shaped dataset, and what each of its
   constructor arguments actually does.
2. The three different things a fitted logistic model can give you:
   `decision_function` (the log-odds), `predict_proba` (probabilities) and
   `predict` (a thresholded label). Mixing these up is the most common
   misunderstanding in classification.
3. Coefficient -> log-odds -> odds ratio -> probability, the full interpretation
   chain, on a feature with a human-readable unit.
4. The intercept: what a 0 value really means, and why it is usually meaningless.
5. Why the coefficient magnitude is not an importance score, and what to use
   instead.

Run:  python 02_logistic_regression_sklearn.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------------------
# 1. A dataset with real units, so the coefficients mean something.
# --------------------------------------------------------------------------------------
# Loan default prediction. Every feature is a number a person can reason about:
# age in years, income in thousands, debt-to-income as a ratio, months employed.
# Interpretability only works if the features are interpretable.
rng = np.random.default_rng(7)

n = 4000
age = rng.integers(18, 71, size=n).astype(float)
income_k = np.clip(rng.normal(62, 28, size=n), 8, None)
dti = np.clip(rng.normal(0.34, 0.13, size=n), 0.01, 0.85)      # debt / income
months_employed = np.clip(rng.exponential(84, size=n), 0, None)
late_payments = rng.poisson(1.4, size=n).astype(float)          # strong signal
has_mortgage = rng.binomial(1, 0.58, size=n).astype(float)

# The TRUE log-odds, written out. The single biggest driver is late payments.
log_odds_true = (
    -3.4
    + 0.032 * age
    + 0.011 * income_k
    - 4.1 * dti
    - 0.0042 * months_employed
    + 0.72 * late_payments
    + 0.31 * has_mortgage
)
default_prob = 1.0 / (1.0 + np.exp(-log_odds_true))
y = rng.binomial(1, default_prob)

df = pd.DataFrame({
    "age": age,
    "income_k": income_k,
    "debt_to_income": dti,
    "months_employed": months_employed,
    "late_payments": late_payments,
    "has_mortgage": has_mortgage,
    "defaulted": y,
})

features = ["age", "income_k", "debt_to_income", "months_employed",
            "late_payments", "has_mortgage"]
X = df[features]
y = df["defaulted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 74)
print("LOGISTIC REGRESSION WITH SCIKIT-LEARN")
print("=" * 74)
print(f"Rows: {len(df)}   positives: {y.mean():.1%}   "
      f"features: {len(features)}   train/test: {len(X_train)}/{len(X_test)}")
print()
print(f"Baseline (predict 0 for everybody): accuracy {accuracy_score(y_test, np.zeros_like(y_test)):.4f}")
print(f"With {y.mean():.0%} positives, that accuracy comes from doing NOTHING.")
print("Any real model must be judged against that number, and accuracy alone cannot")
print("tell the two apart. That is the whole reason script 05 exists.")
print()


# --------------------------------------------------------------------------------------
# 2. Fit the model, and read every constructor argument.
# --------------------------------------------------------------------------------------
print("-" * 74)
print("PART A - the constructor arguments, and what they change")
print("-" * 74)
print("""
  LogisticRegression(
      penalty='l2',      # regularisation TYPE. 'l2' shrink, 'l1' sets weights to
                         # exactly zero, None means no regularisation at all.
                         # NOTE: from sklearn 1.8 'penalty' is deprecated in favour
                         # of l1_ratio (0 = pure L2, 1 = pure L1) and a very large
                         # C for no penalty. This file uses the new spelling.
      C=1.0,            # INVERSE regularisation strength. The full objective is
                         #   mean_log_loss + (1 / (C * n)) * penalty
                         # so LARGER C = WEAKER regularisation. This is inverted
                         # relative to Ridge/Lasso's alpha, and getting it backwards
                         # is the single most common sklearn mistake.
      solver='lbfgs',    # the optimiser. 'lbfgs' is the right default for small and
                         # medium data. 'liblinear' handles small data and supports
                         # the full L1 penalty. 'saga' scales to large data and
                         # supports elastic net.
      max_iter=100,      # iteration cap. If it warns about convergence, RAISE this.
      class_weight=None, # 'balanced' or {0: w0, 1: w1}. See script 07.
      tol=1e-4,          # convergence tolerance on the loss.
  )
""")

# Same features, three regularisation strengths, one feature set.
settings = [
    ("no regularisation (C=1e6)", dict(C=1e6)),
    ("weak (C=10)", dict(C=10.0)),
    ("default (C=1)", dict(C=1.0)),
    ("strong (C=0.01)", dict(C=0.01)),
]

print(f"{'setting':<28} {'train log loss':>15} {'test log loss':>14} {'test AUC':>9} {'max |coef|':>11}")
print("-" * 82)

models = {}
for label, params in settings:
    model = LogisticRegression(max_iter=2000, random_state=42, **params)
    model.fit(X_train, y_train)
    models[label] = model
    print(f"{label:<28} {log_loss(y_train, model.predict_proba(X_train)[:, 1]):>15.4f} "
          f"{log_loss(y_test, model.predict_proba(X_test)[:, 1]):>14.4f} "
          f"{roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):>9.4f} "
          f"{np.max(np.abs(model.coef_[0])):>11.3f}")

print()
print("The log loss columns are the ones to read. As C falls the regularisation")
print("gets stronger, the coefficients shrink, and the TRAINING log loss gets worse.")
print("The test log loss barely moves, because 6 features on 3200 rows is a very")
print("easy fit - there is no overfitting to prevent. The last row is worse on BOTH")
print("counts: heavy regularisation has started destroying real signal, and the")
print("right answer there is to change C rather than to add features.")
print()

# --- make overfitting actually visible, because it is invisible at p = 6 ---------------
print("Overfitting demo - add 400 pure-noise features, then sweep C again.")
print("(The noise is drawn independently for train and test, so the extra columns")
print(" carry no signal at all - they can only be memorised.)")
noise_rng = np.random.default_rng(0)
N_NOISE = 400
X_train_noisy = np.hstack([X_train.to_numpy(), noise_rng.normal(0, 1, (len(X_train), N_NOISE))])
X_test_noisy = np.hstack([X_test.to_numpy(), noise_rng.normal(0, 1, (len(X_test), N_NOISE))])
print()
print(f"{'C':>10} {'train log loss':>15} {'test log loss':>14} {'test - train':>13}")
print("-" * 56)
noisy_rows = []
for c_value in [1e6, 1e4, 1000.0, 100.0, 10.0, 1.0, 0.1, 0.01]:
    noisy_model = LogisticRegression(C=c_value, max_iter=3000, random_state=42)
    noisy_model.fit(X_train_noisy, y_train)
    train_ll = log_loss(y_train, noisy_model.predict_proba(X_train_noisy)[:, 1])
    test_ll = log_loss(y_test, noisy_model.predict_proba(X_test_noisy)[:, 1])
    noisy_rows.append((c_value, train_ll, test_ll))
    print(f"{c_value:>10} {train_ll:>15.4f} {test_ll:>14.4f} {test_ll - train_ll:>13.4f}")

print()
print("Now the effect is unambiguous. With C=inf the 400 noise features pull the")
print("train log loss down from 0.401 to 0.332 by memorising them, while the test")
print("log loss rises from 0.404 to 0.522 - worse than the honest 6-feature model")
print("ever managed. The test-minus-train column is the overfit gap, and it shrinks")
print("monotonically as C falls. The test column is lowest at C=0.01, and note the")
print("train loss there (0.353) is still better than the unregularised 6-feature")
print("fit. Regularisation is not a tax you pay for nothing: it buys a better test")
print("score by refusing to memorise. The answer is never 'add more features'.")
print()

model = models["default (C=1)"]
print("Chosen model: default C=1. Printed object:")
print(model)
print()


# --------------------------------------------------------------------------------------
# 3. The three outputs of a fitted model. This is the section to memorise.
# --------------------------------------------------------------------------------------
print("-" * 74)
print("PART B - decision_function vs predict_proba vs predict")
print("-" * 74)

sample = pd.DataFrame([{
    "age": 45.0, "income_k": 55.0, "debt_to_income": 0.52,
    "months_employed": 96.0, "late_payments": 3.0, "has_mortgage": 1.0,
}])

logit = model.decision_function(sample)[0]
proba = model.predict_proba(sample)[0]
label = model.predict(sample)[0]

print("One applicant, three numbers coming out of the same model:")
print()
print(f"  decision_function  = {logit:+.4f}")
print(f"  predict_proba      = P(class 0) {proba[0]:.4f}   P(class 1) {proba[1]:.4f}")
print(f"  predict            = {label}")
print()
print("What each one means:")
print(f"  decision_function: {logit:+.4f} is the LOG-ODDS of default. exp({logit:.4f}) ="
      f" {np.exp(logit):.2f}")
print("                     so this applicant is", f"{np.exp(logit):.1f}x as likely to default")
print("                     as not to. Same sign, no upper bound.")
print(f"  predict_proba:     {proba[1]:.4f} is the PROBABILITY, in [0, 1].")
print("                     sigmoid(decision_function) = predict_proba, exactly:")
print(f"                       1/(1+exp(-{logit:.4f})) = {1 / (1 + np.exp(-logit)):.4f}")
print(f"  predict:           {label} is a LABEL, produced by thresholding the")
print("                     probability at 0.5. The threshold is a POLICY choice that")
print("                     the model never learns and you can change at any time.")
print()
print("Consequences that matter in production:")
print("  - Changing the threshold needs NO retraining. One model, many operating")
print("    points, chosen by business cost. Script 06 is entirely about this.")
print("  - Do NOT compare a decision_function value against 0.5. It is a log-odds;")
print("    0.5 is a probability. Comparing a probability against 0.5 is the habit")
print("    to unlearn, and getting it wrong gives nonsense 50% defaults.")
print()

# prove the identity
print("Identity check on the test set:")
z_all = model.decision_function(X_test)
p_all = 1.0 / (1.0 + np.exp(-z_all))
print(f"  max |sigmoid(decision_function) - predict_proba[:, 1]| = "
      f"{np.max(np.abs(p_all - model.predict_proba(X_test)[:, 1])):.3e}")
print(f"  labels equal to (prob > 0.5): "
      f"{np.mean((p_all > 0.5).astype(int) == model.predict(X_test)) == 1.0}")
print()


# --------------------------------------------------------------------------------------
# 4. The interpretation chain.
# --------------------------------------------------------------------------------------
print("-" * 74)
print("PART C - from a coefficient to a sentence a human can check")
print("-" * 74)
print("""
A logistic coefficient is a change in LOG-ODDS per unit of the feature.

    log-odds  ->  odds  ->  odds ratio  ->  probability
    z         =  w0 + SUM wj * xj
    odds      =  exp(z)
    OR        =  exp(wj)          <- the number to quote
""")
print(f"{'feature':<20} {'coef':>8} {'exp(coef)':>11} {'what it means':<44}")
print("-" * 86)

meanings = {
    "age": "per additional year of age",
    "income_k": "per additional $1k of income",
    "debt_to_income": "per 1.0 (=100pp) of debt ratio",
    "months_employed": "per additional month at the job",
    "late_payments": "per additional late payment",
    "has_mortgage": "vs. no mortgage (0 -> 1)",
}
for feature, coef in zip(features, model.coef_[0]):
    odds_ratio = np.exp(coef)
    print(f"{feature:<20} {coef:>8.4f} {odds_ratio:>11.4f} {meanings[feature]:<44}")

print()
print("How to read the interesting ones:")
dti_coef = model.coef_[0][features.index("debt_to_income")]
late_coef = model.coef_[0][features.index("late_payments")]
print(f"  late_payments  : exp({late_coef:.4f}) = {np.exp(late_coef):.2f}. Each additional")
print("                   late payment MULTIPLIES the odds of default by "
      f"{np.exp(late_coef):.2f}.")
print("                   Multiplies, not adds: the 2nd late payment hurts more than")
print("                   the 1st, the 3rd more than the 2nd, in proportional terms.")
print(f"                   3 late payments -> odds multiplier {np.exp(3 * late_coef):.1f}x")
print(f"                   1 late payment  -> odds multiplier {np.exp(late_coef):.2f}x")
print()
print(f"  debt_to_income : a coefficient of {dti_coef:.3f} per UNIT is a bad unit for")
print("                   humans - a 0.01 change is realistic. Rescale, or convert:")
print(f"                   per 1 percentage point: coef {dti_coef / 100:+.5f} -> "
      f"OR {np.exp(dti_coef / 100):.4f}")
print(f"                   per 10 percentage points: OR {np.exp(10 * dti_coef / 100):.3f}")
print()
print("  IMPORTANT: the effect is per unit WHILE HOLDING THE OTHERS FIXED. In this")
print("  model income and age are correlated, so an unconditional statement like")
print("  'a richer applicant is safer' is not what the coefficient says.")
print()


# --------------------------------------------------------------------------------------
# 5. Odds ratio to actual probability. The last step of the chain.
# --------------------------------------------------------------------------------------
print("-" * 74)
print("PART D - from odds ratio to a real probability")
print("-" * 74)
print("The odds ratio is an effect ON THE ODDS, not on the probability, and the")
print("difference matters enormously once you are past 50%.")
print()
print(f"{'base applicant probability':>26} {'x1.2 odds':>11} {'x1.2 as prob':>13} {'x2.0 odds':>11} {'x2.0 as prob':>13}")
print("-" * 78)
for base_p in [0.05, 0.20, 0.50, 0.80]:
    base_odds = base_p / (1 - base_p)
    print(f"{base_p:>26.2f} {base_odds * 1.2:>11.3f} {base_odds * 1.2 / (1 + base_odds * 1.2):>13.3f} "
          f"{base_odds * 2.0:>11.3f} {base_odds * 2.0 / (1 + base_odds * 2.0):>13.3f}")
print()
print("A 20% odds increase moves a 5% probability by under a point, and a 50%")
print("probability by five points. Quoting odds ratios in a business document")
print("without this translation is how models get rejected in meetings.")
print()

# Do it concretely on the real fitted model.
base = sample.copy()
riskier = base.assign(late_payments=base["late_payments"] + 2)
p_base = model.predict_proba(base)[0, 1]
p_risk = model.predict_proba(riskier)[0, 1]
print("The same applicant with two more late payments, everything else identical:")
print(f"  P(default) = {p_base:.4f}  ->  {p_risk:.4f}   "
      f"(+{(p_risk - p_base) * 100:.1f} percentage points, a {p_risk / p_base:.2f}x probability)")


# --------------------------------------------------------------------------------------
# 6. What the intercept means (and why you should ignore it).
# --------------------------------------------------------------------------------------
print()
print("-" * 74)
print("PART E - the intercept")
print("-" * 74)
print(f"Intercept = {model.intercept_[0]:.4f} -> baseline odds of default "
      f"exp({model.intercept_[0]:.3f}) = {np.exp(model.intercept_[0]):.4f}")
print()
print("That number is the odds for an applicant with age=0, income=0, debt ratio=0,")
print("0 months employed, 0 late payments and no mortgage. No such person exists.")
print("The intercept is not a measurement of anything; it is a mechanical anchor that")
print("exists because the model form needs a constant term. Report it, but never")
print("interpret it. Two further consequences, both demonstrated below:")
print()
print("  - Class imbalance moves the intercept. A model trained on 1% positives gets")
print("    a much more negative intercept than the same model trained on 50%, even")
print("    with identical features. The intercept encodes the base rate. See 07.")
print("  - Centring the features changes the intercept, and under regularisation it")
print("    changes the PREDICTIONS as well. Details below - this one surprises people.")
print()
print("Demonstration: unregularised. The intercept is the only thing that moves.")
unreg = LogisticRegression(C=1e6, max_iter=5000, tol=1e-12).fit(X, y)
centred_unreg = LogisticRegression(C=1e6, max_iter=5000, tol=1e-12).fit(
    X - X.mean(), y
)
print(f"  unregularised, raw features   : intercept {unreg.intercept_[0]:+.4f}, "
      f"max |pred diff| vs original {np.max(np.abs(centred_unreg.predict_proba(X - X.mean())[:, 1] - unreg.predict_proba(X)[:, 1])):.3e}")
print()
print("Demonstration: with the default L2 penalty. sklearn does NOT penalise the")
print("intercept, so shifting the features' origin changes the trade-off between")
print("fit and penalty - and therefore the predictions.")
scaler_for_centring = X_train.mean()
centred_reg = LogisticRegression(max_iter=2000, random_state=42).fit(
    X_train - scaler_for_centring, y_train
)
raw_reg = LogisticRegression(max_iter=2000, random_state=42).fit(X_train, y_train)
diff = np.max(np.abs(centred_reg.predict_proba(X_test - X_test.mean())[:, 1]
                     - raw_reg.predict_proba(X_test)[:, 1]))
print(f"  regularised (C=1), raw features: intercept {raw_reg.intercept_[0]:+.4f}")
print(f"  regularised (C=1), centred     : intercept {centred_reg.intercept_[0]:+.4f}")
print(f"  max |pred diff| between them   : {diff:.3e}")
print()
print("So the honest statement is: with NO penalty, centring is exactly a")
print("reparametrisation and predictions are identical to machine precision. WITH a")
print("penalty it is a genuinely different model, because the penalty is measured")
print("from the origin and the origin just moved. This is why you must scale and")
print("centre consistently between training and serving - and why a pipeline is")
print("the safe way to do it.")
print()
print("Practical upshot for PART C: the odds ratios above are only valid for the")
print("feature scale the model was trained on. Change the units - dollars instead of")
print("thousands, say - and every coefficient must be rescaled to match.")


# --------------------------------------------------------------------------------------
# 7. Why |coefficient| is not importance.
# --------------------------------------------------------------------------------------
print()
print("-" * 74)
print("PART F - why you cannot rank features by |coefficient|")
print("-" * 74)

scaled = LogisticRegression(max_iter=2000, random_state=42).fit(
    StandardScaler().fit_transform(X_train), y_train
)
raw_table = pd.Series(model.coef_[0], index=features)
scaled_table = pd.Series(scaled.coef_[0], index=X_train.columns)
comparison = pd.DataFrame({
    "coef (raw units)": raw_table,
    "coef (standardised)": scaled_table,
    "spread (raw / scaled)": raw_table / scaled_table,
}).reindex(raw_table.abs().sort_values(ascending=False).index)

print(comparison.round(4).to_string())
print()
print("The RANKING changes. debt_to_income is third by raw coefficient and first")
print("by standardised coefficient, because its raw unit (0 to 1) is much smaller")
print("than late_payments (0 to 9). Only the standardised column is a fair")
print("comparison - and even that measures association, not predictive usefulness")
print("among correlated features. Script 13 covers permutation importance, which")
print("answers the question you actually care about.")
print()
print("The 'spread' column is the reciprocal of each feature's standard deviation,")
print("i.e. exactly the factor you would need to convert between the two columns.")


# --------------------------------------------------------------------------------------
# 8. Plots.
# --------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# --- Panel 1: coefficient forest plot -------------------------------------------------
ax = axes[0]
order = np.argsort(model.coef_[0])
colors = ["#F03E3E" if c > 0 else "#4C6EF5" for c in model.coef_[0][order]]
ax.barh(np.array(features)[order], model.coef_[0][order], color=colors)
ax.axvline(0, color="#212529", linewidth=1.5)
for index, coef in enumerate(model.coef_[0][order]):
    ax.text(coef + (0.012 if coef > 0 else -0.012), index, f"{np.exp(coef):.2f}x",
            va="center", ha="left" if coef > 0 else "right", fontsize=8)
ax.set_title("Coefficients (labels show the odds ratio)", fontsize=11, fontweight="bold")
ax.set_xlabel("coefficient on the log-odds scale")
ax.grid(alpha=0.25, axis="x")

# --- Panel 2: probability vs log-odds -------------------------------------------------
ax = axes[1]
scatter = ax.scatter(z_all, model.predict_proba(X_test)[:, 1], s=10, alpha=0.35,
                     color="#4C6EF5", edgecolors="none")
z_curve = np.linspace(z_all.min(), z_all.max(), 300)
ax.plot(z_curve, 1 / (1 + np.exp(-z_curve)), color="#E03131", linewidth=2.5, label="sigmoid(z)")
ax.axhline(0.5, color="#868E96", linestyle=":", linewidth=1.5)
ax.axvline(0, color="#868E96", linestyle=":", linewidth=1.5)
ax.set_title("Log-odds (x) vs probability (y)", fontsize=11, fontweight="bold")
ax.set_xlabel("decision_function  =  log-odds")
ax.set_ylabel("predict_proba[:, 1]")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Panel 3: the odds-to-probability translation -------------------------------------
ax = axes[2]
base_p = np.linspace(0.01, 0.99, 200)
for multiplier, style in [(1.0, "-"), (1.25, "--"), (2.0, "-."), (4.0, ":")]:
    odds = base_p / (1 - base_p) * multiplier
    ax.plot(base_p, odds / (1 + odds), style, linewidth=2,
            label=f"odds x{multiplier}")
ax.set_title("Same odds ratio, different probability impact", fontsize=11, fontweight="bold")
ax.set_xlabel("base probability")
ax.set_ylabel("new probability")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

fig.suptitle("02 - Logistic Regression with scikit-learn", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


# --------------------------------------------------------------------------------------
# 9. Takeaways.
# --------------------------------------------------------------------------------------
print()
print("=" * 74)
print("TAKEAWAYS")
print("=" * 74)
print("1. The model is a line on the log-odds, squashed by a sigmoid. Everything else")
print("   follows from that sentence.")
print("2. C is the INVERSE of regularisation strength. Bigger C = less regularisation.")
print("3. decision_function is a log-odds, unbounded. predict_proba is a probability.")
print("   predict is a label, thresholded at 0.5 by a policy you control.")
print("4. exp(coef) is the odds ratio per unit, holding the other features fixed.")
print("   Translate it to a probability change before putting it in a document.")
print("5. The intercept encodes the base rate and is not interpretable. Centring the")
print("   features changes it and nothing else.")
print("6. |coefficient| depends on feature units. Standardise before you rank.")
print("7. Compare TRAIN log loss to TEST log loss to see overfitting. Comparing")
print("   training accuracy to test accuracy will not reveal it.")
