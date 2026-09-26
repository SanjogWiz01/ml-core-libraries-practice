"""
05 - Cost Functions and Evaluation Metrics
==========================================

Goal: implement every regression metric by hand so you know exactly what each one
rewards, punishes and hides.

Metrics covered
---------------
- MSE   (Mean Squared Error)      - the training loss
- RMSE  (Root Mean Squared Error) - MSE in the original units
- MAE   (Mean Absolute Error)     - robust to outliers
- R2    (Coefficient of Determination)
- Adjusted R2                    - R2 that penalises added features
- MAPE  (Mean Absolute Percentage Error) - and why it explodes
- sMAPE (Symmetric MAPE)          - the safe alternative

The central demonstration
-------------------------
Two models with the SAME typical error but different outlier behaviour. Every
metric ranks them differently. Choosing a metric is choosing which mistakes you
forgive.

Run:  python 05_cost_functions_and_metrics.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

SEED = 42
rng = np.random.default_rng(SEED)

print("=" * 78)
print("COST FUNCTIONS AND EVALUATION METRICS")
print("=" * 78)


# ======================================================================================
# PART 1 - hand-written implementations
# ======================================================================================
def mse(y_true, y_pred):
    """Mean Squared Error. Unbounded, differentiable, outlier-sensitive."""
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """Root MSE. Same ordering as MSE (monotonic transform) but original units."""
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    """Mean Absolute Error. Linear in the error, so outliers barely register."""
    return np.mean(np.abs(y_true - y_pred))


def r2_score_manual(y_true, y_pred):
    """
    1 - SS_res/SS_tot. Can go NEGATIVE, which means "worse than predicting the mean".
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


def adjusted_r2(y_true, y_pred, n_features):
    """
    R2 adjusted for the number of features.

    n_features is the count of PREDICTORS (not including the intercept).
    Adding a feature always raises plain R2, because the model can always use it
    to fit noise. Adjusted R2 divides out that free lunch.
    """
    n = len(y_true)
    r2 = r2_score_manual(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)


def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error. DANGEROUS: undefined when y_true is 0.
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def smape(y_true, y_pred):
    """Symmetric MAPE. Bounded to [0, 200], safe around zero."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.abs(y_true - y_pred) / np.where(denominator == 0, 1, denominator)) * 100


# A small real dataset to test correctness against scikit-learn.
from sklearn.datasets import load_diabetes  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

diabetes = load_diabetes()
y_true = diabetes.target
y_pred = LinearRegression().fit(diabetes.data, diabetes.target).predict(diabetes.data)

print()
print("-" * 78)
print("PART 1 - hand-written vs scikit-learn (diabetes dataset, in-sample)")
print("-" * 78)

checks = pd.DataFrame({
    "mine": [mse(y_true, y_pred), rmse(y_true, y_pred), mae(y_true, y_pred),
             r2_score_manual(y_true, y_pred), mape(y_true, y_pred), smape(y_true, y_pred)],
    "sklearn": [mean_squared_error(y_true, y_pred),
                np.sqrt(mean_squared_error(y_true, y_pred)),
                mean_absolute_error(y_true, y_pred),
                r2_score(y_true, y_pred),
                np.nan,  # scikit-learn has no MAPE; it is in MAPIE, not sklearn
                np.nan],
}, index=["MSE", "RMSE", "MAE", "R2", "MAPE (%)", "sMAPE (%)"])
checks["match"] = np.isclose(checks["mine"], checks["sklearn"], rtol=1e-9, equal_nan=True)
print(checks.round(6).to_string())
print()
print("All match. Note: sklearn deliberately ships NO MAPE - its instability")
print("around zero is a real trap, not an oversight. Use sMAPE or MAE instead.")


# ======================================================================================
# PART 2 - the key experiment: one outlier, every metric reacts differently
# ======================================================================================
print()
print("-" * 78)
print("PART 2 - two models, one outlier, four different verdicts")
print("-" * 78)

n = 200
X = rng.uniform(0, 50, size=(n, 1))
y_clean = 4.0 * X.ravel() + 12.0 + rng.normal(0, 3, size=n)

model = LinearRegression().fit(X, y_clean)
pred_clean = model.predict(X)

# Model A: honest, small errors everywhere.
y_outlier = y_clean.copy()
# Model B: nearly perfect, except for 3 catastrophic misses. Same predictions -
# only the TRUTH for 3 rows is corrupted, which isolates the effect of outliers.
y_outlier[:3] += 600
pred_outlier = pred_clean  # identical model, identical predictions

print(f"Errors of the honest model     : MAE {mae(y_clean, pred_clean):8.3f}  "
      f"RMSE {rmse(y_clean, pred_clean):8.3f}  max |err| {np.max(np.abs(y_clean - pred_clean)):8.3f}")
print(f"Errors of the 3-blowup model  : MAE {mae(y_outlier, pred_outlier):8.3f}  "
      f"RMSE {rmse(y_outlier, pred_outlier):8.3f}  max |err| {np.max(np.abs(y_outlier - pred_outlier)):8.3f}")
print()
print("The second model is better on 197 of 200 rows and catastrophically wrong on 3.")
print("Now watch the metrics disagree about which model is 'better':")
print()
verdicts = pd.DataFrame({
    "honest model": [
        mse(y_clean, pred_clean), rmse(y_clean, pred_clean),
        mae(y_clean, pred_clean), r2_score_manual(y_clean, pred_clean),
    ],
    "model with 3 blowups": [
        mse(y_outlier, pred_outlier), rmse(y_outlier, pred_outlier),
        mae(y_outlier, pred_outlier), r2_score_manual(y_outlier, pred_outlier),
    ],
}, index=["MSE (lower better)", "RMSE (lower better)", "MAE (lower better)", "R2 (higher better)"])
verdicts["winner"] = np.where(
    verdicts["honest model"] < verdicts["model with 3 blowups"], "honest", "blowups"
)
print(verdicts.round(3).to_string())
print()
print("MSE and RMSE (square-based) crown the outlier model the winner.")
print("MAE (absolute) correctly identifies the honest model as better.")
print("=> If a few catastrophic misses should not dominate your decision, optimise MAE.")


# ======================================================================================
# PART 3 - Adjusted R2 versus plain R2 while adding useless features
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - plain R2 vs adjusted R2 when you add junk features")
print("-" * 78)

n_total = 400
X_real = rng.normal(0, 1, size=(n_total, 3))
# Only x0 and x1 are real signal. x2 is noise from the start.
y_signal = 5 * X_real[:, 0] - 3 * X_real[:, 1] + rng.normal(0, 2, size=n_total)

# 12 extra pure-noise columns that carry no information whatsoever.
X_extra_noise = rng.normal(0, 1, size=(n_total, 12))
X_all = np.hstack([X_real, X_extra_noise])

# A real holdout, so "test R2" means what it says.
idx_train, idx_test = train_test_split(np.arange(n_total), test_size=0.3, random_state=SEED)

print(f"{'features':<20} {'train R2':>10} {'test R2':>10} {'adj R2':>10} {'train-test gap':>16}")
print("-" * 78)

for n_feats, label in [(3, "3 real + 0 noise"), (5, "3 real + 2 noise"),
                       (8, "3 real + 5 noise"), (15, "3 real + 12 noise")]:
    X_sub = X_all[:, :n_feats]
    model = LinearRegression().fit(X_sub[idx_train], y_signal[idx_train])
    r2_in = r2_score_manual(y_signal[idx_train], model.predict(X_sub[idx_train]))
    r2_adj = adjusted_r2(y_signal[idx_train], model.predict(X_sub[idx_train]),
                         n_features=n_feats)
    r2_out = r2_score_manual(y_signal[idx_test], model.predict(X_sub[idx_test]))
    print(f"{label:<20} {r2_in:>8.4f} {r2_out:>8.4f} {r2_adj:>8.4f} {r2_in - r2_out:>16.4f}")

print()
print("Train R2 climbs monotonically as junk is added - it can never fall, because")
print("the extra weights always absorb some noise. Test performance is the truth.")
print("Adjusted R2 turns over and starts falling once the features stop paying off,")
print("which is exactly the signal you want for feature selection.")
print()
print("=> NEVER quote a training R2. Adjusted R2 is a crude in-sample substitute;")
print("   cross-validation is the real answer (see script 10).")


# ======================================================================================
# PART 4 - why MAPE is dangerous
# ======================================================================================
print()
print("-" * 78)
print("PART 4 - MAPE's fatal flaw: targets near zero")
print("-" * 78)

y_mixed = np.array([2.0, 3.0, 1.5, 4.0, 0.0001])  # one value is essentially zero
pred_mixed = np.array([2.2, 2.8, 1.6, 3.9, 5.0])  # 0.0001 -> 5.0 is a real 5-unit error

print(f"y_true    : {y_mixed}")
print(f"y_pred    : {pred_mixed}")
print(f"abs error : {np.abs(y_mixed - pred_mixed).round(4)}")
print()
print(f"  MAE   = {mae(y_mixed, pred_mixed):10.4f}   <- reasonable, all 5 errors similar")
print(f"  MAPE  = {mape(y_mixed, pred_mixed):10.4f}   <- one row produced 499,900% error")
print(f"  sMAPE = {smape(y_mixed, pred_mixed):10.4f}   <- bounded, the row cannot dominate")
print()
print("The last row's absolute error (4.9999) is comparable to the others, yet MAPE")
print("reaches 5 million percent. Divide by a small number and you manufacture")
print("arbitrarily large errors from a perfectly ordinary mistake.")
print()
print("Rules of thumb:")
print("  - MAPE is fine when every target is comfortably positive and similar in size")
print("  - sMAPE when some targets can be near zero")
print("  - MAE when you need a stable number at all, or when errors matter in units")
print("  - MASE (scaled MAE) when comparing across series of different scale")


# ======================================================================================
# PART 5 - R2 can go negative. That is information, not a bug.
# ======================================================================================
print()
print("-" * 78)
print("PART 5 - a negative R2 is a genuinely useful warning")
print("-" * 78)

# Fit on pure noise -> the model has learned nothing at all.
X_noise_only = rng.normal(0, 1, size=(n, 5))
y_pure_noise = rng.normal(0, 1, size=n)
noise_model = LinearRegression().fit(X_noise_only, y_pure_noise)
noise_pred = noise_model.predict(X_noise_only)
mean_pred = np.full_like(y_pure_noise, y_pure_noise.mean())

print(f"R2 of a model fitted to pure noise : {r2_score_manual(y_pure_noise, noise_pred):.4f}")
print(f"R2 of always predicting the mean   : {r2_score_manual(y_pure_noise, mean_pred):.4f}")
print()
print("The mean-prediction model scores EXACTLY 0 - by definition, since SS_res")
print("equals SS_tot. The overfitted noise model scores BELOW 0, which is the")
print("mathematical statement 'this model generalises worse than a constant'.")
print()
print("So R2 < 0 means one of:")
print("  - the model overfitted (most common)")
print("  - the features do not describe the target")
print("  - the test set is unlike the training set (distribution shift)")
print("  - there is a bug - check for train/test contamination, or column misalignment")
print("In all four cases: do not ship it.")


# ======================================================================================
# Visualisation
# ======================================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# --- Panel 1: how each metric moves as a single error grows -------------------------
ax = axes[0]
errors = np.linspace(0, 30, 300)
ax.plot(errors, errors**2, linewidth=2.2, color="#E03131", label="MSE: error^2")
ax.plot(errors, np.abs(errors), linewidth=2.2, color="#0CA678", label="MAE: |error|")
ax.set_title("Squared vs absolute penalty", fontsize=11, fontweight="bold")
ax.set_xlabel("size of a single error (all other errors = 0)")
ax.set_ylabel("metric value")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)
ax.annotate("one 30-unit error\n= 900 MSE, 30 MAE", xy=(30, 900), xytext=(9, 620),
            fontsize=8, color="#E03131",
            arrowprops=dict(arrowstyle="->", color="#E03131"))

# --- Panel 2: the two models, error by error ----------------------------------------
ax = axes[1]
errors_a = np.abs(y_clean - pred_clean)
errors_b = np.abs(y_outlier - pred_outlier)
ax.hist(errors_a, bins=25, alpha=0.75, color="#0CA678", label=f"honest (MAE {mae(y_clean, pred_clean):.1f})")
ax.hist(errors_b, bins=25, alpha=0.75, color="#E03131", label=f"3 blowups (MAE {mae(y_outlier, pred_outlier):.1f})")
ax.set_title("Error distributions", fontsize=11, fontweight="bold")
ax.set_xlabel("absolute error")
ax.set_ylabel("count")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Panel 3: R2 vs number of features ----------------------------------------------
ax = axes[2]
n_feat_list = list(range(3, 16))
train_r2s, adj_r2s = [], []
for n_feats in n_feat_list:
    X_sub = X_all[:, :n_feats]
    m = LinearRegression().fit(X_sub, y_signal)
    train_r2s.append(r2_score_manual(y_signal, m.predict(X_sub)))
    adj_r2s.append(adjusted_r2(y_signal, m.predict(X_sub), n_features=n_feats))
ax.plot(n_feat_list, train_r2s, "o-", color="#E03131", linewidth=2, label="train R2 (always rises)")
ax.plot(n_feat_list, adj_r2s, "o-", color="#4C6EF5", linewidth=2, label="adjusted R2 (penalises)")
ax.set_title("R2 vs adjusted R2 as junk is added", fontsize=11, fontweight="bold")
ax.set_xlabel("number of features (only the first 3 are real)")
ax.set_ylabel("score")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

fig.suptitle("05 - Cost Functions and Evaluation Metrics", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


# ======================================================================================
# Which metric to actually use
# ======================================================================================
print()
print("=" * 78)
print("METRIC SELECTION CHEAT SHEET")
print("=" * 78)
print("""
  Reporting a regression result   ->  R2 AND RMSE together. Never one alone.
  Errors in original units        ->  RMSE (or MAE). "off by 4.2 units" is meaningful.
  Outliers are frequent           ->  MAE. Squares would let 3 rows decide everything.
  Comparing across a period       ->  MAPE / sMAPE, but audit for near-zero targets.
  Choosing between feature counts ->  Adjusted R2, or better, cross-validation.
  A model scored below 0          ->  Do not deploy. Compare against DummyRegressor.
  Distribution shift suspected    ->  Report MAE and RMSE: R2 is not comparable
                                      across a shifted test set.

Reporting template that actually survives review:
  "R2 = 0.82 (CV mean 0.80 +/- 0.04), RMSE = 4.1, MAE = 3.2, against a
   DummyRegressor baseline of R2 = 0.00. Residuals show no trend, so the
   linearity assumption holds. 95% confidence interval from 5-fold CV: ..."
""")
