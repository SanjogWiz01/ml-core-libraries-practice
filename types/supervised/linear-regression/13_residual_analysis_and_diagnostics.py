"""
13 - Residual Analysis and Diagnostics
======================================

Goal: read the residuals of a fitted regression and know which assumption is
broken, so you know what to change.

What residuals are
------------------
    residual = y_true - y_predicted

If the model were perfect, residuals would be random noise: symmetric, no pattern,
constant spread, no outliers. In practice they almost never are, and each pattern
points at a specific problem.

The five checks, and what each failure means
-------------------------------------------
| Check                  | Symptom                  | Assumption broken | Fix |
|------------------------|--------------------------|-------------------|-----|
| Residuals vs fitted    | curve / U / S shape      | LINEARITY         | polynomial, splines, transform y, GBM |
| Residuals vs fitted    | funnel (fan)             | CONSTANT VARIANCE | log1p(y), weighted fit, robust model |
| QQ plot                | heavy tails, S-shape     | NORMAL residuals  | robust regression, transform, bootstrap CIs |
| Histogram of residuals | skew or multiple modes   | functional form   | feature engineering, splines |
| Residuals vs index     | runs, waves              | INDEPENDENCE      | differencing, time-series split, GLS |
| Cook's distance        | a few points dominating  | no outliers       | remove, winsorise, Huber, RANSAC |

The crucial distinction
-----------------------
Linearity, independence and constant variance affect the COEFFICIENTS and the
PREDICTIONS. Normality and absence of outliers affect only the CONFIDENCE
INTERVALS and p-values. If you only care about predictions, assumption 5 and 6
are survivable. If you want to claim "this coefficient is significant", they are not.

Run:  python 13_residual_analysis_and_diagnostics.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

SEED = 42
rng = np.random.default_rng(SEED)

print("=" * 78)
print("RESIDUAL ANALYSIS AND DIAGNOSTICS")
print("=" * 78)


# ======================================================================================
# The diagnostics themselves
# ======================================================================================
def breusch_pagan_like(residuals, fitted):
    """
    A simple, interpretable stand-in for the formal Breusch-Pagan test.

    Formal BP regresses squared residuals on the fitted values and tests whether
    the slope is significantly non-zero. We do the regression and report the R2
    of that auxiliary regression - a high value means the residual spread grows
    with the prediction, i.e. heteroscedasticity. No p-value, but the same idea,
    readable without a statistics library.
    """
    squared = residuals**2
    design = np.column_stack([np.ones_like(fitted), fitted])
    beta, *_ = np.linalg.lstsq(design, squared, rcond=None)
    predicted = design @ beta
    ss_res = np.sum((squared - predicted) ** 2)
    ss_tot = np.sum((squared - squared.mean()) ** 2)
    return 1 - ss_res / ss_tot, beta[1]


def durbin_watson(residuals):
    """
    DW = SUM (e_t - e_{t-1})^2 / SUM e_t^2

    ~2.0  -> no autocorrelation
    <2.0  -> positive autocorrelation (the classic "errors run in runs" case)
    >2.0  -> negative autocorrelation (errors alternate, usually an artefact)

    This measures autocorrelation in the ORDER of the rows. It is only meaningful
    if that order carries meaning - i.e. if the rows are a time series.
    """
    diffs = np.diff(residuals)
    return np.sum(diffs**2) / np.sum(residuals**2)


def durbin_watson_breusch_pagan(residuals):
    """The packaged version of the same test."""
    from statsmodels.stats.stattools import durbin_watson as dw
    from statsmodels.stats.diagnostic import het_breuschpagan as bp
    dw_stat = dw(residuals)
    lm_stat, lm_p, f_stat, f_p = bp(residuals, np.column_stack([np.ones(len(residuals))]))
    return dw_stat, lm_p, f_p


# ======================================================================================
# PART 1 - build four datasets, each with exactly one broken assumption
# ======================================================================================
print()
print("-" * 78)
print("PART 1 - four models, four different problems, caught by residuals")
print("-" * 78)

n = 400
X = rng.uniform(1, 20, size=(n, 1)).ravel()

# (a) Correct specification - linear, homoscedastic, normal errors.
y_a = 3.0 * X + 10.0 + rng.normal(0, 3, size=n)
# (b) NON-LINEAR truth fitted with a straight line.
y_b = 0.08 * X**3 - X**2 + 5 * X + 20 + rng.normal(0, 3, size=n)
# (c) HETEROSCEDASTIC: the spread grows with the level of x.
y_c = 2.0 * X + 5.0 + rng.normal(0, 1, size=n) * (0.15 * X)
# (d) A few gross outliers, plus the linear truth.
y_d = 2.0 * X + 5.0 + rng.normal(0, 3, size=n)
y_d[:12] += 220

cases = {
    "a: correct model": y_a,
    "b: non-linear truth": y_b,
    "c: heteroscedastic": y_c,
    "d: with outliers": y_d,
}

print(f"{'case':<24} {'R2':>8} {'RMSE':>8} {'res mean':>10} {'res std':>9} {'DW':>7} {'max|r|':>8}")
print("-" * 78)

fits = {}
for name, y_case in cases.items():
    Xtr, Xte, ytr, yte = train_test_split(X.reshape(-1, 1), y_case, test_size=0.25,
                                          random_state=SEED)
    model = LinearRegression().fit(Xtr, ytr)
    # Residuals on the TRAINING set, which is where patterns are visible.
    resid_train = ytr - model.predict(Xtr)
    fits[name] = {
        "model": model, "Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte,
        "resid_train": resid_train,
        "r2": model.score(Xte, yte),
        "rmse": np.sqrt(np.mean((yte - model.predict(Xte)) ** 2)),
    }
    f = fits[name]
    print(f"{name:<24} {f['r2']:>8.4f} {f['rmse']:>8.3f} {resid_train.mean():>10.3f} "
          f"{resid_train.std():>9.3f} {durbin_watson(resid_train):>7.2f} "
          f"{np.max(np.abs(resid_train)):>8.1f}")

print()
print("Every model has a plausible-looking R2. None of those R2 values tells you")
print("what is actually wrong. The residual plots do.")


# ======================================================================================
# PART 2 - the quantitative diagnostics for each case
# ======================================================================================
print()
print("-" * 78)
print("PART 2 - quantifying each assumption violation")
print("-" * 78)

for name in cases:
    resid = fits[name]["resid_train"]
    fitted = fits[name]["model"].predict(fits[name]["Xtr"])
    bp_r2, bp_slope = breusch_pagan_like(resid, fitted)
    print(f"\n{name}")
    print(f"  residual mean              : {resid.mean():+.3f}  (should be ~0)")
    print(f"  residual skewness          : {pd.Series(resid).skew():+.3f}  (should be ~0)")
    print(f"  residual kurtosis          : {pd.Series(resid).kurtosis():+.3f}  (should be ~0 for normal, 3 is the normal value)")
    print(f"  heteroscedasticity aux R2  : {bp_r2:.4f}  (should be near 0; high = funnel shape)")
    print(f"  spread at low vs high fits : {resid[fitted < np.median(fitted)].std():.3f} vs "
          f"{resid[fitted > np.median(fitted)].std():.3f}  (a big ratio = non-constant variance)")
    print(f"  Durbin-Watson              : {durbin_watson(resid):.2f}  (~2 = independent)")
    print(f"  fraction of |resid| > 3sd  : {np.mean(np.abs(resid) > 3 * resid.std()):.4f}  "
          f"(normal expectation: {2 * 0.0027:.4f})")

print()
print("The 'fraction beyond 3 sd' is the quickest outlier screen: for genuinely")
print("normal residuals about 0.54% of them should exceed 3 standard deviations.")
print("Case d is far above that, which is the numerical signature of contamination.")


# ======================================================================================
# PART 3 - the fixes
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - the fix for each case, and the measured improvement")
print("-" * 78)

fixes = {}

# (a) needs nothing
fixes["a: correct model"] = ("nothing to do", y_a, X)

# (b) the fix needs x, x^2 and x^3 - built per-branch below from the 1-D base column
fixes["b: non-linear truth"] = ("add x^2 and x^3 terms", y_b, X)

# (c) log1p the target, or fit with weights
fixes["c: heteroscedastic"] = ("log1p the target, invert at the end", y_c, X)

# (d) drop the outliers
mask_out = np.abs(y_d - (2 * X + 5)) < 60
fixes["d: with outliers"] = ("remove the 12 contaminated rows", y_d[mask_out], X[mask_out])

print(f"{'case':<24} {'original R2':>12} {'fixed R2':>10} {'method':>42}")
print("-" * 78)


def powers(x_1d, degree):
    """Design matrix of [x, x^2, ..., x^degree] from a 1-D array."""
    return np.column_stack([x_1d**p for p in range(1, degree + 1)])


for name, (method, y_fixed, X_fixed) in fixes.items():
    # X_fixed is always the 1-D base column, so this is safe regardless of case.
    X_fixed_2d = np.asarray(X_fixed).reshape(-1, 1)
    Xtr, Xte, ytr, yte = train_test_split(X_fixed_2d, y_fixed,
                                          test_size=0.25, random_state=SEED)
    if name.startswith("b:"):
        m = LinearRegression().fit(powers(Xtr.ravel(), 3), ytr)
        pred = m.predict(powers(Xte.ravel(), 3))
    elif name.startswith("c:"):
        m = LinearRegression().fit(Xtr, np.log1p(np.clip(ytr, 0, None)))
        pred = np.expm1(m.predict(Xte))
    else:
        m = LinearRegression().fit(Xtr, ytr)
        pred = m.predict(Xte)
    fixed_r2 = 1 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2)
    print(f"{name:<24} {fits[name]['r2']:>12.4f} {fixed_r2:>10.4f} {method:>42}")

print()
print("Case (c) deserves a note. Taking a log of the target is the single most")
print("useful transform in regression, because it does two things at once:")
print("  - it compresses the right tail, often removing heteroscedasticity")
print("  - it turns multiplicative effects into additive ones")
print("But it introduces RETRANSMFORMATION BIAS: E[exp(f(x))] != exp(E[f(x)]), so")
print("inverting predictions systematically underestimates the mean. The standard")
print("correction for a log-normal model is to multiply the inverse by")
print("exp(sigma^2 / 2). Measure sigma from the residuals of the log-scale fit:")
print("")

# Demonstrate the retransformation bias and the smearing correction.
resid_log = np.log1p(np.clip(y_c, 0, None)) - LinearRegression().fit(
    X.reshape(-1, 1), np.log1p(np.clip(y_c, 0, None))).predict(X.reshape(-1, 1))
sigma2 = np.var(resid_log)
Xtr, Xte, ytr, yte = train_test_split(X.reshape(-1, 1), y_c, test_size=0.25,
                                      random_state=SEED)
m_log = LinearRegression().fit(Xtr, np.log1p(np.clip(ytr, 0, None)))
naive = np.expm1(m_log.predict(Xte))
corrected = np.expm1(m_log.predict(Xte)) * np.exp(np.mean(resid_log**2) / 2)

print(f"sigma^2 of log-scale residuals : {np.mean(resid_log**2):.4f}")
print(f"naive inverse transform   MAE  : {np.mean(np.abs(yte - naive)):.3f}")
print(f"smearing-corrected        MAE  : {np.mean(np.abs(yte - corrected)):.3f}")
print()
print("The correction closes roughly half the gap. Two other standard options:")
print("  - Duan's smearing estimator: pred * mean(exp(residuals))")
print("  - quantile regression at the 0.5 level, which needs no correction at all")


# ======================================================================================
# PART 4 - QQ plot, done by hand
# ======================================================================================
print()
print("-" * 78)
print("PART 4 - the normal Q-Q plot, built manually")
print("-" * 78)
print("A Q-Q plot answers one question: are these residuals normally distributed?")
print()
print("  1. Sort the residuals.")
print("  2. For each rank i of n, the expected standard-normal quantile is")
print("     z_i = Phi^-1((i - 0.5) / n)   (the plotting position)")
print("  3. Plot residuals on the y-axis, z_i on the x-axis.")
print("  4. Perfect normality = a straight line.")
print()
print("Theoretical quantiles are NOT sample quantiles here - if you sorted the")
print("data and used those as the x-values you would always get a straight line,")
print("which would make the plot useless.")

from scipy import stats  # noqa: E402

print(f"\n{'case':<24} {'skew':>7} {'kurtosis':>10} {'Shapiro p':>11} {'normal?':>9}")
print("-" * 66)
for name in cases:
    resid = fits[name]["resid_train"]
    stat, p_value = stats.shapiro(resid)
    print(f"{name:<24} {pd.Series(resid).skew():>7.3f} {pd.Series(resid).kurtosis():>10.3f} "
          f"{p_value:>11.4f} {'yes' if p_value > 0.05 else 'NO':>9}")
print()
print("Caveat worth stating out loud: with n = 300, Shapiro-Wilk rejects normality")
print("for almost anything, including genuinely normal data. Treat it as a")
print("sanity check, not a verdict. Look at the PLOT and at practical magnitudes.")


# ======================================================================================
# PART 5 - influence diagnostics
# ======================================================================================
print()
print("-" * 78)
print("PART 5 - which individual rows are dragging the fit?")
print("-" * 78)

X_fit = X.reshape(-1, 1)
y_fit = y_d
model_d = LinearRegression().fit(X_fit, y_fit)
resid_d = y_fit - model_d.predict(X_fit)

# Hat values: how much leverage does each row have? A row near the centre of x
# but far in y is influential. A row at the edge of x is high-leverage.
leverage = np.diag(X_fit @ np.linalg.pinv(X_fit.T @ X_fit) @ X_fit.T)
# Studentised residuals: residuals scaled by the model's own uncertainty AT THAT ROW.
standardised = resid_d / (resid_d.std() * np.sqrt(1 - leverage))
# Cook's distance: combines leverage and residual size into one influence number.
cooks = (standardised**2 / 2) * (leverage / (1 - leverage))

summary = pd.DataFrame({
    "x": X.ravel(),
    "y": y_d,
    "residual": resid_d,
    "leverage": leverage,
    "studentised": standardised,
    "cooks_d": cooks,
}).sort_values("cooks_d", ascending=False)

print("Top 8 most influential rows:")
print(summary.head(8).round(3).to_string(index=False))
print()
print("Interpretation:")
print(f"  leverage        : mean {leverage.mean():.4f}, 2/n = {2 / n:.4f}. Anything")
print("                    well above 2/n is an outlier in the FEATURE space -")
print("                    usually an x near the edge of the range.")
print(f"  studentised > 3 : {int(np.sum(np.abs(standardised) > 3))} rows are more than 3")
print("                    sigma off, after accounting for their own leverage.")
print(f"  Cook's D > 4/n  : {int(np.sum(cooks > 4 / n))} rows exceed the common threshold 4/n.")
print("                    These are the rows to actually go and investigate.")
print()
print("The proper procedure is never 'delete the outliers'. It is:")
print("  1. Look at the influential rows. Is it a data entry error, a different")
print("     population, or a legitimate rare case?")
print("  2. Data entry error -> fix it. A unit mistake (5000 instead of 50.00) is")
print("     the single most common real cause.")
print("  3. Different population -> fit a model with an indicator, or model them")
print("     separately.")
print("  4. Legitimate extreme -> keep it, and use Huber or RANSAC (script 15).")
print()
print(f"Effect of removing the {int(np.sum(cooks > 4 / n))} flagged rows:")
keep = cooks <= 4 / n
m_clean = LinearRegression().fit(X_fit[keep], y_fit[keep])
m_all = LinearRegression().fit(X_fit, y_fit)
print(f"  with outliers   : slope {m_all.coef_[0]:.4f}  intercept {m_all.intercept_:.3f}")
print(f"  without them    : slope {m_clean.coef_[0]:.4f}  intercept {m_clean.intercept_:.3f}")
print(f"  truth           : slope 2.0000  intercept 5.000")
print()
print("Twelve corrupted rows out of 400 - 3% of the data - moved the slope by")
print(f"{abs(m_all.coef_[0] - 2.0) / 2.0 * 100:.1f}%. That is the fragility of OLS.")


# ======================================================================================
# Visualisation
# ======================================================================================
fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
axes = axes.ravel()

colors = {"a: correct model": "#0CA678", "b: non-linear truth": "#E03131",
          "c: heteroscedastic": "#F59F00", "d: with outliers": "#7048E8"}

for idx, (name, f) in enumerate(fits.items()):
    resid = f["resid_train"]
    fitted = f["model"].predict(f["Xtr"])
    color = colors[name]

    # --- residuals vs fitted: the primary diagnostic --------------------------------
    ax = axes[idx]
    ax.scatter(fitted, resid, s=18, alpha=0.5, color=color, edgecolors="none")
    ax.axhline(0, color="#212529", linestyle="--", linewidth=1.8)
    # LOWESS-style smoother to make the shape obvious to the eye.
    order = np.argsort(fitted)
    smooth = pd.Series(resid[order]).rolling(25, center=True, min_periods=5).mean()
    ax.plot(fitted[order], smooth, color="#212529", linewidth=2.5, label="smoothed")

    # Flag the threshold that matters for this case.
    if name.startswith("b:"):
        ax.set_title("B: U shape -> NON-LINEARITY\nmodel form is wrong", fontsize=10,
                     fontweight="bold", color="#E03131")
    elif name.startswith("c:"):
        ax.set_title("C: funnel -> HETEROSCEDASTICITY\nerror grows with prediction",
                     fontsize=10, fontweight="bold", color="#F59F00")
    elif name.startswith("d:"):
        ax.set_title("D: isolated points -> OUTLIERS\n12 rows dominate the fit",
                     fontsize=10, fontweight="bold", color="#7048E8")
    else:
        ax.set_title("A: shapeless cloud -> CORRECT\nassumptions hold", fontsize=10,
                     fontweight="bold", color="#0CA678")

    ax.set_xlabel("fitted value")
    ax.set_ylabel("residual")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)

# --- Panel 4: Q-Q plot for all four ----------------------------------------------------
ax = axes[4]
for name, f in fits.items():
    resid = np.sort(f["resid_train"])
    n_obs = len(resid)
    theoretical = stats.norm.ppf((np.arange(1, n_obs + 1) - 0.5) / n_obs)
    standardised_resid = (resid - resid.mean()) / resid.std()
    ax.scatter(theoretical, standardised_resid, s=5, alpha=0.4, color=colors[name],
               edgecolors="none", label=name)
lims = [min(theoretical), max(theoretical)]
ax.plot(lims, lims, color="#212529", linewidth=2, linestyle="--", label="perfect normal")
ax.set_title("Normal Q-Q: points should follow the dashed line", fontsize=10, fontweight="bold")
ax.set_xlabel("theoretical normal quantile")
ax.set_ylabel("standardised residual")
ax.legend(fontsize=7, frameon=False, loc="upper left")
ax.grid(alpha=0.25)

# --- Panel 5: influence ----------------------------------------------------------------
ax = axes[5]
ax.scatter(leverage, standardised, c=cooks, s=45, cmap="plasma", edgecolors="black",
           linewidths=0.4)
# Standard 4/n and 1.0 Cook's distance contours.
n_total = len(y_d)
grid_h = np.linspace(0, max(leverage.max() * 1.1, 0.01), 100)
for d_limit, style in [(4 / n_total, "--"), (1.0, ":")]:
    ax.plot(grid_h, np.sqrt(d_limit * 2 * (1 - grid_h) / np.maximum(grid_h, 1e-6)),
            color="#E03131", linestyle=style, linewidth=1.8)
    ax.plot(grid_h, -np.sqrt(d_limit * 2 * (1 - grid_h) / np.maximum(grid_h, 1e-6)),
            color="#E03131", linestyle=style, linewidth=1.8)
ax.set_xlim(0, max(leverage.max() * 1.1, 0.01))
ax.set_title("Leverage vs studentised residual\n(red contours: Cook's D = 4/n and 1.0)",
             fontsize=10, fontweight="bold")
ax.set_xlabel("leverage (hat value)")
ax.set_ylabel("studentised residual")
ax.grid(alpha=0.25)

fig.suptitle("13 - Residual Analysis and Diagnostics", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
print("1. A good R2 tells you nothing about whether the model is correctly specified.")
print("   Residuals tell you everything.")
print("2. Curve in residuals vs fitted        -> wrong functional form (bias)")
print("   Funnel in residuals vs fitted       -> non-constant variance")
print("   Isolated points far from zero       -> outliers or a data entry error")
print("   Waves when plotted against row order-> dependence between observations")
print("3. log1p(y) is the highest-value single fix: it tames the right tail and")
print("   linearises multiplicative effects. Remember the retransformation bias.")
print("4. Cook's distance, leverage and studentised residuals locate the rows worth")
print("   investigating. Investigate before removing - never just delete.")
print("5. Normality of residuals matters for INFERENCE (p-values, CIs), not for the")
print("   predictions. Bootstrap if you need trustworthy intervals.")
print("6. Shapiro-Wilk with n in the hundreds rejects normality almost always.")
print("   Judge normality by the plot and by practical magnitude, not by the p-value.")
