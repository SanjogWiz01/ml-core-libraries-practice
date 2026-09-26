"""
15 - Outliers and Robust Regression
===================================

Goal: understand why a handful of bad rows can wreck an ordinary least squares fit,
and learn four different ways to fix it.

Why OLS is so fragile
----------------------
OLS minimises the SUM OF SQUARED residuals. A single row 100 units off contributes
10,000 to the loss; the other 199 rows contribute almost nothing in comparison. So
the optimiser will happily move the entire line to accommodate that one row,
sacrificing accuracy everywhere else. The cost of squared error is unbounded, and
one unbounded term dominates everything.

Breakdown point: the fraction of the data a model tolerates being arbitrarily
corrupted before its estimate stops converging to the truth.

| Estimator              | Breakdown point | Notes                                  |
|------------------------|-----------------|----------------------------------------|
| OLS / LinearRegression | 0%             | Moves arbitrarily for a single bad row |
| Theil-Sen (Sen slope)  | ~29%           | Median-based slope                     |
| RANSAC                 | ~30-50%        | Explicitly separates inliers/outliers  |
| HuberRegressor         | n/a (soft)     | Quadratic near 0, linear in the tails  |
| QuantileRegressor      | n/a (soft)     | Optimises a conditional quantile        |
| MAD / trimmed estimator| ~20-40%        | Discards the extremes outright         |

The four fixes, and when each is right
--------------------------------------
1. **HuberRegressor** - smooth, keeps every row, best when outliers are a SMALL
   fraction and you still want a coefficient-like answer. One hyperparameter, `epsilon`.
2. **RANSACRegressor** - explicitly finds inliers by sampling, best when outliers
   are a LARGE fraction, or when you want to inspect and hard-reject bad rows.
3. **QuantileRegressor (median)** - the least squares of robust methods. Naturally
   ignores outliers because the median of the errors is minimised, not their sum.
4. **Winsorising / capping** - domain-appropriate when you KNOW the valid range
   (a price cannot be negative; an age cannot be 250) and want to keep OLS.

Run:  python 15_outliers_and_robust_regression.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import (
    HuberRegressor,
    LinearRegression,
    QuantileRegressor,
    RANSACRegressor,
    TheilSenRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

SEED = 42
rng = np.random.default_rng(SEED)

print("=" * 78)
print("OUTLIERS AND ROBUST REGRESSION")
print("=" * 78)


# ======================================================================================
# PART 1 - how much damage can a few rows do?
# ======================================================================================
print()
print("-" * 78)
print("PART 1 - the damage, measured")
print("-" * 78)

# Clean, well-behaved data with a known truth.
n = 200
X_clean = rng.uniform(0, 100, size=(n, 1))
TRUE_SLOPE, TRUE_INTERCEPT = 2.5, 10.0
y_clean = TRUE_SLOPE * X_clean.ravel() + TRUE_INTERCEPT + rng.normal(0, 3, size=n)

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.3, random_state=SEED
)
clean_model = LinearRegression().fit(X_train, y_train)
print(f"Clean data, ordinary OLS:")
print(f"  slope     = {clean_model.coef_[0]:.4f}   (truth {TRUE_SLOPE})")
print(f"  intercept = {clean_model.intercept_:.4f}   (truth {TRUE_INTERCEPT})")
print(f"  test R2   = {clean_model.score(X_test, y_test):.4f}")
print()

# Now contaminate the TRAINING set only. The test set stays clean, which is
# exactly what happens in reality: bad historical rows poison the model, and the
# damage only shows up on future data.
print("Adding outliers to the TRAINING set only (the test set stays clean):")
print()
print(f"{'% corrupted':>12} {'n_outliers':>12} {'slope':>10} {'slope error':>13} "
      f"{'intercept':>11} {'test R2':>9}")
print("-" * 72)

for fraction in [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]:
    n_outliers = int(n * 0.7 * fraction)
    X_c = X_train.copy()
    y_c = y_train.copy()
    if n_outliers:
        # Corrupt by pushing the target far off the true line.
        positions = rng.choice(len(y_c), n_outliers, replace=False)
        y_c[positions] += rng.choice([-1, 1], n_outliers) * rng.uniform(120, 400, n_outliers)

    model = LinearRegression().fit(X_c, y_c)
    slope_error = abs(model.coef_[0] - TRUE_SLOPE) / TRUE_SLOPE * 100
    print(f"{fraction * 100:>11.0f}% {n_outliers:>12} {model.coef_[0]:>10.3f} "
          f"{slope_error:>12.1f}% {model.intercept_:>11.1f} "
          f"{model.score(X_test, y_test):>9.4f}")

print()
print("Two percent of the training rows, corrupted, moves the slope by more than 20%")
print("and halves the test score. That is the price of squared error with no bound.")
print("The intercept suffers even more, because outliers cluster in x and drag the")
print("line vertically as well as angularly.")


# ======================================================================================
# PART 2 - the same data, six different estimators
# ======================================================================================
print()
print("-" * 78)
print("PART 2 - six estimators on identical contaminated data")
print("-" * 78)

# Build a fixed contaminated training set so every method sees the same thing.
X_c = X_train.copy()
y_c = y_train.copy()
n_outliers = 12
outlier_positions = rng.choice(len(y_c), n_outliers, replace=False)
y_c[outlier_positions] += rng.choice([-1, 1], n_outliers) * rng.uniform(150, 350, n_outliers)

print(f"12 of {len(y_c)} training rows corrupted ({n_outliers / len(y_c) * 100:.1f}%).")
print(f"Truth: slope {TRUE_SLOPE}, intercept {TRUE_INTERCEPT}. Test set is CLEAN.")
print()
print(f"{'estimator':<24} {'slope':>9} {'intercept':>11} {'test R2':>9} {'test MAE':>10} "
      f"{'training n_used':>17}")
print("-" * 96)

estimators = {
    "OLS (LinearRegression)": LinearRegression(),
    "Huber (eps=1.35)": HuberRegressor(epsilon=1.35, max_iter=500),
    "Huber (eps=1.10)": HuberRegressor(epsilon=1.10, max_iter=500),
    "RANSAC": RANSACRegressor(random_state=SEED, residual_threshold=8.0),
    "Theil-Sen": TheilSenRegressor(random_state=SEED),
    "Quantile (median)": QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs"),
}

results = {}
for name, model in estimators.items():
    model.fit(X_c, y_c)
    pred = model.predict(X_test)
    # RANSAC does not forward coef_/intercept_; its refitted inlier model does.
    inner = model if hasattr(model, "coef_") else model.estimator_
    slope = float(np.ravel(inner.coef_)[0])
    intercept = float(np.ravel(inner.intercept_)[0])
    # RANSAC knows which rows it rejected - a genuinely useful output.
    if hasattr(model, "inlier_mask_"):
        n_used = f"{int(model.inlier_mask_.sum())} inliers"
    else:
        n_used = "all rows (soft weighting)"
    results[name] = {"slope": slope, "intercept": intercept,
                     "r2": r2_score(y_test, pred),
                     "mae": mean_absolute_error(y_test, pred)}
    print(f"{name:<24} {slope:>9.3f} {intercept:>11.2f} "
          f"{results[name]['r2']:>9.4f} {results[name]['mae']:>10.3f} {n_used:>17}")

print("-" * 96)
print()
print("OLS recovers a slope nowhere near 2.5. Every robust estimator gets close.")
print()
print("Reading the 'n_used' column - this is the key design difference:")
print()
print("  Huber and Quantile use EVERY row, but weight them non-linearly. A gross")
print("  outlier still influences the fit, just very little. Robust, smooth, no")
print("  hard decisions, but you cannot list the rows it distrusted.")
print()
print("  RANSAC draws a hard line: inliers, and everything else discarded. That")
print("  gives you `model.estimator_.sample_weight_` and `inlier_mask_`, so you can")
print("  go and inspect every rejected row. Far more useful operationally.")


# ======================================================================================
# PART 3 - Huber's loss function, and what epsilon does
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - Huber's loss: quadratic in the middle, linear in the tails")
print("-" * 78)
print("Huber replaces the square with a function that stops growing quadratically:")
print()
print("              0.5 * r^2           if |r| <= epsilon      <- same as squared error")
print("    rho(r) = epsilon * (|r| - 0.5*epsilon)   otherwise    <- LINEAR growth")
print()
print("Small residuals behave exactly as in least squares. Large residuals cost")
print("proportionally instead of quadratically, so a single huge point cannot")
print("dominate the objective. epsilon sets the width of the quadratic zone.")
print()

epsilons = [1.0, 1.1, 1.35, 2.0, 5.0]
print(f"{'epsilon':>9} {'slope':>9} {'intercept':>11} {'test R2':>9} {'interpretation'}")
print("-" * 88)
for eps in epsilons:
    huber = HuberRegressor(epsilon=eps, max_iter=1000).fit(X_c, y_c)
    pred = huber.predict(X_test)
    if eps <= 1.0:
        note = "the minimum: hardest line against outliers"
    elif eps < 1.5:
        note = "the standard default (95% efficient, normal errors)"
    elif eps < 3:
        note = "closer to OLS, more sensitive to outliers"
    else:
        note = "nearly ordinary least squares"
    print(f"{eps:>9.2f} {huber.coef_[0]:>9.3f} {huber.intercept_:>11.2f} "
          f"{r2_score(y_test, pred):>9.4f} {note}")

print()
print("epsilon = 1.35 is the classical default, chosen because the resulting")
print("estimator is 95% as efficient as OLS when the errors really are normal, while")
print("being dramatically less sensitive to contamination. Lower epsilon means")
print("suspicion; higher means trust.")
print()
print("Huber uses an ITERATIVE reweighting scheme internally, so the diagnostic you")
print("want is the final sample weight. Outliers get a small weight, not zero:")
print()
huber_final = HuberRegressor(epsilon=1.35, max_iter=1000).fit(X_c, y_c)
weights = np.ravel(huber_final.coef_)
# Recompute what the effective weights look like via one refit on clean data only.
clean_huber = HuberRegressor(epsilon=1.35, max_iter=1000).fit(X_train, y_train)
print(f"  coefficients on contaminated data : {np.round(huber_final.coef_, 3)}")
print(f"  coefficients on clean data        : {np.round(clean_huber.coef_, 3)}")
print(f"  truth                             : [{TRUE_SLOPE}]")
print()
print("The two are nearly identical. Huber has effectively ignored the 12 bad rows")
print("without ever being told which ones they were.")


# ======================================================================================
# PART 4 - RANSAC in detail
# ======================================================================================
print()
print("-" * 78)
print("PART 4 - RANSAC: identifying WHICH rows are bad")
print("-" * 78)
print("RANSAC works by repetition:")
print("  1. Sample a tiny random subset (2 points for a line)")
print("  2. Fit a model to it")
print("  3. Measure how many other points are within residual_threshold of it")
print("  4. Keep the candidate with the most inliers")
print("  5. Refit using all the inliers")
print()

ransac = RANSACRegressor(
    estimator=LinearRegression(),
    residual_threshold=8.0,
    min_samples=0.3,          # need only 30% of the data to agree before accepting
    max_trials=5000,
    random_state=SEED,
)
ransac.fit(X_c, y_c)

inlier_mask = ransac.inlier_mask_
n_inliers = int(inlier_mask.sum())
outlier_idx = np.where(~inlier_mask)[0]
truth_outliers = set(outlier_positions.tolist())

print(f"Rows RANSAC accepted as inliers  : {n_inliers} of {len(y_c)}")
print(f"Rows RANSAC rejected             : {len(outlier_idx)}")
print(f"Rows actually corrupted          : {n_outliers}")
correctly_caught = len(set(outlier_idx.tolist()) & truth_outliers)
false_positives = len(set(outlier_idx.tolist()) - truth_outliers)
missed = len(truth_outliers - set(outlier_idx.tolist()))
print()
print(f"  correctly identified as bad : {correctly_caught}/{n_outliers}")
print(f"  good rows wrongly rejected   : {false_positives}")
print(f"  bad rows let through         : {missed}")
print()
print(f"Slope from all data (OLS)         : {LinearRegression().fit(X_c, y_c).coef_[0]:.3f}")
print(f"Slope from RANSAC inliers only    : {np.ravel(ransac.estimator_.coef_)[0]:.3f}")
print(f"Truth                             : {TRUE_SLOPE}")
print()
print("The rejected rows are the operationally valuable output. In production you")
print("would join them back to your source system and find out what happened - a")
print("decimal-point error, a unit change, a duplicate transaction, a refunded")
print("order that should be negative. RANSAC tells you WHERE to look.")
print()
print("Rejected rows (x, corrupted y, original y):")
rejected = pd.DataFrame({
    "row_position": outlier_idx,
    "x": X_c.ravel()[outlier_idx],
    "corrupted_y": y_c[outlier_idx],
    "original_y": y_train[outlier_idx],
    "was_really_bad": [i in truth_outliers for i in outlier_idx],
}).sort_values("row_position")
print(rejected.head(12).round(2).to_string(index=False))


# ======================================================================================
# PART 5 - quantile regression: a different question, robustly answered
# ======================================================================================
print()
print("-" * 78)
print("PART 5 - quantile regression and the CLEAN vs DIRTY comparison")
print("-" * 78)
print("Quantile regression asks a different question:")
print()
print("  least squares     : 'what is the line through the MIDDLE of the points?'")
print("  median regression : 'what line has HALF the points above and half below?'")
print("  0.9 quantile      : 'what line has 90% of points BELOW it?'")
print()
print("Because it minimises the pinball (quantile) loss, the median fit is")
print("mathematically immune to outliers - their magnitude does not matter, only")
print("which side they fall on. This is a genuinely different model, not a fix.")
print()

print(f"{'quantile':>10} {'slope':>9} {'intercept':>11} {'test MAE':>10} {'interpretation'}")
print("-" * 84)
for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
    q_model = QuantileRegressor(quantile=q, alpha=0.0, solver="highs")
    q_model.fit(X_train, y_train)
    pred = q_model.predict(X_test)
    if q == 0.5:
        note = "the median line - the robust 'typical' fit"
    elif q < 0.5:
        note = "lower bound of the conditional distribution"
    else:
        note = "upper bound - use for risk / upper-quantile forecasts"
    print(f"{q:>10.2f} {q_model.coef_[0]:>9.3f} {q_model.intercept_:>11.2f} "
          f"{mean_absolute_error(y_test, pred):>10.3f} {note}")

print()
print("Fitted on the CLEAN training set, so the slope is stable across quantiles.")
print("Refit on contaminated data and watch the tails move while the median holds:")
print()
for q in [0.1, 0.5, 0.9]:
    q_model = QuantileRegressor(quantile=q, alpha=0.0, solver="highs")
    q_model.fit(X_c, y_c)
    print(f"  quantile {q:.1f} on contaminated data: slope {q_model.coef_[0]:>8.3f}  "
          f"intercept {q_model.intercept_:>9.2f}")
print()
print("The 0.1 and 0.9 fits are dragged by the corrupted rows; the 0.5 fit barely")
print("moves. That is the pinball loss working as designed.")


# ======================================================================================
# PART 6 - when to just clean the data
# ======================================================================================
print()
print("-" * 78)
print("PART 6 - winsorising: sometimes the right answer is a domain rule")
print("-" * 78)
print("A robust estimator is the right tool when outliers are LEGITIMATE - a rare")
print("but real event. It is the wrong tool when the value is simply IMPOSSIBLE.")
print("Then the answer is a domain constraint, not a statistical one.")
print()
print("Examples of each:")
print("  legitimate, keep the row : a viral product's sales, a record temperature")
print("  impossible, correct it   : a negative price, an age of 250, a 30-hour day")
print("  impossible, reject it    : a quantity with no units, a test record")
print()
print("Winsorising caps values at a chosen quantile instead of deleting them:")
print("  [ 12, 15, 11, 14, 9000 ]  ->  cap at the 95th percentile")
print("  [ 12, 15, 11, 14, 9000 ]  ->  becomes roughly [ 12, 15, 11, 14, 14.3 ]")
print("  Keeps the sample size, which deletion does not. Use it when you have a")
print("  defensible bound, and document the bound.")
print()
print("In code:")
print("""
    upper = y_train.quantile(0.995)          # compute the bound on TRAIN only
    y_winsorised = y_train.clip(upper=upper) # cap, do not delete
    print(f"capped {int((y_train > upper).sum())} rows at {upper:.1f}")

    # Or a hard domain rule, which is better when you have one:
    df.loc[df["age"] > 120, "age"] = np.nan   # then impute
    assert (df["price"] >= 0).all()            # or drop the impossible rows
""")

# Demonstrate winsorising on the contaminated data.
upper_bound = np.quantile(y_c, 0.98)
lower_bound = np.quantile(y_c, 0.02)
y_winsorised = np.clip(y_c, lower_bound, upper_bound)
winsor_model = LinearRegression().fit(X_c, y_winsorised)
print(f"Bounds from the 2nd and 98th percentiles: [{lower_bound:.1f}, {upper_bound:.1f}]")
print(f"Rows capped: {int(np.sum((y_c < lower_bound) | (y_c > upper_bound)))} of {len(y_c)}")
print()
print(f"{'method':<34} {'slope':>9} {'intercept':>11} {'test R2':>9}")
print("-" * 68)
for name, model in [
    ("OLS on raw contaminated data", LinearRegression().fit(X_c, y_c)),
    ("OLS on winsorised data", winsor_model),
    ("Huber on raw contaminated data", HuberRegressor(epsilon=1.35, max_iter=1000).fit(X_c, y_c)),
    ("RANSAC on raw contaminated data", ransac),
]:
    pred = model.predict(X_test)
    inner = model if hasattr(model, "coef_") else model.estimator_
    print(f"{name:<34} {float(np.ravel(inner.coef_)[0]):>9.3f} "
          f"{float(np.ravel(inner.intercept_)[0]):>11.2f} {r2_score(y_test, pred):>9.4f}")

print()
print("Winsorising works here, but it is a blunt instrument: it moves every")
print("extreme value to the same number, which invents data. Prefer it only when a")
print("domain bound justifies the cap. Huber and RANSAC achieve the same robustness")
print("without fabricating values.")


# ======================================================================================
# PART 7 - scaling and robust regression together
# ======================================================================================
print()
print("-" * 78)
print("PART 7 - RobustScaler: the preprocessing half of the story")
print("-" * 78)
print("Outliers distort the StandardScaler too, not just the model:")
print()
print("  StandardScaler uses mean and standard deviation. A single huge value drags")
print("  the mean toward itself and inflates the standard deviation for EVERY column,")
print("  so the other features get squashed into a tiny range and the model cannot")
print("  resolve them. One bad row degrades the whole model, not just its own term.")
print()
print("  RobustScaler uses the median and the interquartile range instead, neither of")
print("  which is affected by extreme values.")
print()
y_with_outlier = np.append(y_train, 5000.0)
print(f"{'scaler':<16} {'mean after 1 huge value':>26} {'std after 1 huge value':>25}")
print("-" * 70)
print(f"{'StandardScaler':<16} {StandardScaler().fit_transform(y_with_outlier.reshape(-1, 1)).mean():>26.4f}"
      f" {StandardScaler().fit_transform(y_with_outlier.reshape(-1, 1)).std():>25.4f}")
print(f"{'RobustScaler':<16} {RobustScaler().fit_transform(y_with_outlier.reshape(-1, 1)).mean():>26.4f}"
      f" {RobustScaler().fit_transform(y_with_outlier.reshape(-1, 1)).std():>25.4f}")
print(f"{'raw (no scaling)':<16} {y_with_outlier.mean():>26.4f} {y_with_outlier.std():>25.4f}")
print()
print("Compare the mean column: StandardScaler tracks the outlier, RobustScaler")
print("barely moves. That is the whole argument for it.")


# ======================================================================================
# Visualisation
# ======================================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# --- Plot 1: the damage ---------------------------------------------------------------
ax = axes[0, 0]
ax.scatter(X_train, y_train, s=30, alpha=0.6, color="#0CA678", edgecolors="none",
           label="clean rows")
ax.scatter(X_c[outlier_positions], y_c[outlier_positions], s=90, color="#E03131",
           marker="x", linewidths=2.5, label="corrupted rows")
grid = np.linspace(0, 100, 100)
ax.plot(grid, TRUE_SLOPE * grid + TRUE_INTERCEPT, color="#212529", linewidth=2.5,
        linestyle="--", label="truth")
ols_line = LinearRegression().fit(X_c, y_c)
ax.plot(grid, ols_line.coef_[0] * grid + ols_line.intercept_, color="#E03131",
        linewidth=2.5, label=f"OLS (slope {ols_line.coef_[0]:.2f})")
huber_line = HuberRegressor(epsilon=1.35, max_iter=1000).fit(X_c, y_c)
ax.plot(grid, np.ravel(huber_line.coef_)[0] * grid + huber_line.intercept_,
        color="#7048E8", linewidth=2.5, label=f"Huber (slope {np.ravel(huber_line.coef_)[0]:.2f})")
ax.set_title("12 corrupted rows tilt the whole OLS line", fontsize=11, fontweight="bold")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Plot 2: slope error by estimator ------------------------------------------------
ax = axes[0, 1]
names = list(results)
errors = [abs(results[n]["slope"] - TRUE_SLOPE) for n in names]
colors_bar = ["#E03131" if e > 1.0 else "#0CA678" for e in errors]
ax.barh(range(len(names)), errors, color=colors_bar)
ax.axvline(1.0, color="#212529", linestyle="--", linewidth=2,
           label="threshold: 1.0 = useless")
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8)
ax.invert_yaxis()
ax.set_title("Absolute slope error (truth = 2.5)", fontsize=11, fontweight="bold")
ax.set_xlabel("|slope - truth|")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25, axis="x")

# --- Plot 3: loss functions compared --------------------------------------------------
ax = axes[1, 0]
residual_range = np.linspace(-200, 200, 500)
eps = 1.35
squared_loss = 0.5 * residual_range**2
absolute_loss = np.abs(residual_range)
huber_loss = np.where(np.abs(residual_range) <= eps,
                      0.5 * residual_range**2,
                      eps * (np.abs(residual_range) - 0.5 * eps))
pinball_loss = np.maximum(0.5 * residual_range, 0)  # quantile 0.5
ax.plot(residual_range, squared_loss / 1000, linewidth=2.2, color="#E03131",
        label="squared (OLS)")
ax.plot(residual_range, absolute_loss / 10, linewidth=2.2, color="#4C6EF5",
        label="absolute (L1) / 10")
ax.plot(residual_range, huber_loss / 10, linewidth=2.8, color="#0CA678",
        label="Huber / 10")
ax.set_ylim(0, 25)
ax.axvline(-100, color="#868E96", linestyle=":", linewidth=1.5)
ax.axvline(100, color="#868E96", linestyle=":", linewidth=1.5)
ax.annotate("one 100-unit error\ncosts 5000 in OLS", xy=(100, 5), xytext=(-190, 14),
            fontsize=8, color="#E03131",
            arrowprops=dict(arrowstyle="->", color="#E03131"))
ax.set_title("Loss functions: what each one charges for a big mistake", fontsize=11,
             fontweight="bold")
ax.set_xlabel("residual (all / 10 for visual comparison)")
ax.set_ylabel("loss")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Plot 4: Huber epsilon sweep -----------------------------------------------------
ax = axes[1, 1]
eps_sweep = np.logspace(0, 2.5, 40)  # scikit-learn requires epsilon >= 1.0
swept_slopes = [HuberRegressor(epsilon=e, max_iter=1000).fit(X_c, y_c).coef_[0]
                for e in eps_sweep]
swept_r2 = [r2_score(y_test, HuberRegressor(epsilon=e, max_iter=1000)
                     .fit(X_c, y_c).predict(X_test)) for e in eps_sweep]
ax.plot(eps_sweep, swept_slopes, color="#4C6EF5", linewidth=2.2, label="slope")
ax.axhline(TRUE_SLOPE, color="#0CA678", linestyle="--", linewidth=2, label="true slope")
ax.axhline(2.5, color="#E03131", linewidth=2, label="OLS slope on this data")
ax2 = ax.twinx()
ax2.plot(eps_sweep, swept_r2, color="#7048E8", linewidth=2, linestyle=":", label="test R2")
ax2.set_ylabel("test R2", color="#7048E8")
ax2.tick_params(axis="y", labelcolor="#7048E8")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("Huber: small epsilon is robust, large epsilon is just OLS", fontsize=11,
             fontweight="bold")
ax.set_xlabel("epsilon")
ax.set_ylabel("estimated slope (log scale)")
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.grid(alpha=0.25)

fig.suptitle("15 - Outliers and Robust Regression", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
print("1. OLS has a 0% breakdown point: ONE arbitrarily corrupted row can move the")
print("   fit arbitrarily far. Squared error has no upper bound, so one big term")
print("   dominates the whole objective.")
print("2. 2% corrupted training rows cost over 20% of the slope and half the test R2.")
print("   Always inspect residuals for isolated extreme points.")
print("3. HuberRegressor: smooth, uses every row, no hard decisions. Best when")
print("   outliers are a small fraction. epsilon=1.35 is the standard default.")
print("4. RANSACRegressor: hard inlier/outlier split. Best when outliers are a large")
print("   fraction, or when you need the REJECTED ROWS to go and investigate.")
print("5. QuantileRegressor at 0.5 is the median - a genuinely different model, and")
print("   the right choice when you want upper/lower prediction bounds.")
print("6. If the value is IMPOSSIBLE rather than rare, fix the data. Winsorising at")
print("   a domain-justified bound beats any robust estimator, because it does not")
print("   invent a replacement value.")
print("7. Use RobustScaler when outliers are present, so the preprocessing is not")
print("   distorted by the same rows that would distort the model.")
