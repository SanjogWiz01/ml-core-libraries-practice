"""
06 - Bias, Variance and Diagnosing Overfitting
==============================================

Goal: watch the bias-variance tradeoff happen in front of you, and learn the three
diagnostic shapes that tell you what to do next.

The core idea
-------------
    Expected Test Error = Bias^2 + Variance + Irreducible Noise

- **Bias**    - systematically wrong. The model class cannot even represent the truth.
- **Variance**- wildly unstable. The model chases the noise in this particular sample.
- **Noise**   - the part you will never explain no matter what.

Linear regression has low variance (a few parameters, stable) but can have high
bias (a straight line cannot curve). Adding flexibility moves you along the
tradeoff. Model capacity is the dial; learning rate is a different knob entirely.

The three diagnostic shapes
---------------------------
| Residual plot looks like | Diagnosis                | Cure                    |
|--------------------------|--------------------------|-------------------------|
| Clear curve / U / arc    | HIGH BIAS (underfit)     | more features, poly, GBM |
| Random cloud, low R2     | insufficient features   | feature engineering     |
| Great on train, bad on test | HIGH VARIANCE (overfit) | regularise, select, more data |

Run:  python 06_bias_variance_diagnosis.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

SEED = 42
rng = np.random.default_rng(SEED)

print("=" * 78)
print("BIAS, VARIANCE AND OVERFITTING")
print("=" * 78)


# ======================================================================================
# PART 1 - the noise ceiling
# ======================================================================================
print()
print("-" * 78)
print("PART 1 - first, establish the ceiling that noise imposes")
print("-" * 78)

# The true relationship is deliberately NOT a straight line.
def true_function(x):
    return 0.5 * x ** 2 + 2.0 * x - 30.0


NOISE_STD = 8.0  # irreducible error
n_samples = 400
X = rng.uniform(-6, 6, size=n_samples)
y = true_function(X) + rng.normal(0, NOISE_STD, size=n_samples)

print(f"Data: {n_samples} points, x in [{X.min():.1f}, {X.max():.1f}]")
print(f"True function  : 0.5*x^2 + 2*x - 30   <- curved, so a straight line cannot fit it")
print(f"Noise std      : {NOISE_STD}")
print()
print("No model can score R2 = 1 on this data, because the noise is unlearnable.")
print("Knowing the noise floor stops you chasing an impossible target:")
print(f"  best possible R2 ~ 1 - (NOISE_STD^2 / variance(y))")
print(f"  variance(y) ~ {np.var(y):.1f}  ->  ceiling R2 ~ {1 - NOISE_STD**2 / np.var(y):.3f}")
print("Any model near that number has extracted all the available signal.")
print("This is the single most useful calibration step in a regression project.")


# ======================================================================================
# PART 2 - the bias-variance curve as model complexity increases
# ======================================================================================
X_poly = X.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.3, random_state=SEED
)

print()
print("-" * 78)
print("PART 2 - complexity sweep: degree 1 to 15")
print("-" * 78)
print(f"{'degree':>7} {'#params':>9} {'train R2':>10} {'test R2':>10} {'train RMSE':>12} "
      f"{'test RMSE':>11} {'gap':>8} {'verdict':>22}")
print("-" * 78)

sweep = []
for degree in range(1, 16):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_p = poly.fit_transform(X_train)
    X_test_p = poly.transform(X_test)

    model = LinearRegression().fit(X_train_p, y_train)
    train_pred = model.predict(X_train_p)
    test_pred = model.predict(X_test_p)

    train_r2 = model.score(X_train_p, y_train)
    test_r2 = model.score(X_test_p, y_test)
    train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
    test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))
    gap = train_r2 - test_r2

    if train_r2 < 0.85 and test_r2 < 0.85:
        verdict = "HIGH BIAS (underfit)"
    elif gap > 0.15:
        verdict = "HIGH VARIANCE (overfit)"
    else:
        verdict = "just about right"

    sweep.append({
        "degree": degree,
        "n_params": model.coef_.size,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "gap": gap,
        "verdict": verdict,
    })
    print(f"{degree:>7} {model.coef_.size:>9} {train_r2:>10.4f} {test_r2:>10.4f} "
          f"{train_rmse:>12.3f} {test_rmse:>11.3f} {gap:>8.4f} {verdict:>22}")

sweep_df = pd.DataFrame(sweep)
best_idx = sweep_df["test_r2"].idxmax()
print("-" * 78)
print(f"Best test R2 at degree {int(sweep_df.loc[best_idx, 'degree'])} "
      f"(R2 = {sweep_df.loc[best_idx, 'test_r2']:.4f})")
print()
print("Read the table top to bottom:")
print("  - degree 1-2 : train and test BOTH poor -> the model class is too rigid.")
print("                 This is bias, and it only goes away with more flexibility.")
print("  - middle     : test R2 peaks, gap is small -> the sweet spot. Stop here.")
print("  - degree 10+ : train R2 -> 1.00 while test R2 collapses. Textbook overfitting.")
print("                 The extra weights are memorising individual noise points.")


# ======================================================================================
# PART 3 - what overfitting actually looks like in the coefficients
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - the same sweep, from the coefficient side")
print("-" * 78)

for degree in [1, 2, 5, 12]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_p = poly.fit_transform(X_train)
    X_test_p = poly.transform(X_test)
    model = LinearRegression().fit(X_train_p, y_train)
    labels = poly.get_feature_names_out(["x"])

    print(f"\ndegree {degree}: {model.coef_.size} coefficients, test R2 = {model.score(X_test_p, y_test):.4f}")
    coef_df = pd.DataFrame({
        "term": labels,
        "coefficient": model.coef_.round(2),
        "abs": np.abs(model.coef_),
    }).sort_values("abs", ascending=False)
    print(coef_df.head(6).to_string(index=False))
    total_abs = np.abs(model.coef_).sum()
    print(f"  sum |coefficients| = {total_abs:.1f}   <- grows fast as variance grows")

print()
print("At degree 12 there are 13 parameters fitted to 280 training points. The")
print("coefficients are enormous and flip sign wildly between different seeds.")
print("That instability is variance made visible, and it is why a coefficient you")
print("cannot quote with confidence in a high-degree model is a coefficient you")
print("should regularise away (script 12).")


# ======================================================================================
# PART 4 - average over many random splits to see the true behaviour
# ======================================================================================
print()
print("-" * 78)
print("PART 4 - the same degrees, averaged over 25 random splits")
print("-" * 78)
print("A single split is one sample from a noisy distribution. Averaging reveals")
print("the actual curve instead of a random fluctuation.")
print()

rows = []
for degree in [1, 2, 3, 4, 6, 8, 12]:
    train_scores, test_scores = [], []
    for split_seed in range(25):
        X_tr, X_te, y_tr, y_te = train_test_split(X_poly, y, test_size=0.3,
                                                  random_state=split_seed)
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_tr_p = poly.fit_transform(X_tr)
        X_te_p = poly.transform(X_te)
        m = LinearRegression().fit(X_tr_p, y_tr)
        train_scores.append(m.score(X_tr_p, y_tr))
        test_scores.append(m.score(X_te_p, y_te))
    rows.append({
        "degree": degree,
        "train_mean": np.mean(train_scores),
        "train_std": np.std(train_scores),
        "test_mean": np.mean(test_scores),
        "test_std": np.std(test_scores),
    })
    print(f"  degree {degree:>2}:  train R2 {np.mean(train_scores):.4f} +/- {np.std(train_scores):.4f}"
          f"   test R2 {np.mean(test_scores):.4f} +/- {np.std(test_scores):.4f}")

avg_df = pd.DataFrame(rows)
best = avg_df.loc[avg_df["test_mean"].idxmax()]
print()
print(f"Averaged over 25 splits, the best degree is {int(best['degree'])} "
      f"(test R2 = {best['test_mean']:.4f} +/- {best['test_std']:.4f}).")
print("Note the standard deviations: they ARE the variance of the estimate, and")
print("they grow with degree. Reporting a single-split number hides this entirely.")


# ======================================================================================
# PART 5 - the three residual shapes, side by side
# ======================================================================================
print()
print("-" * 78)
print("PART 5 - the three residual shapes and what each one is telling you")
print("-" * 78)

y_pred_line = LinearRegression().fit(X_train, y_train).predict(X_train)

poly2 = PolynomialFeatures(degree=2, include_bias=False)
X_train_p2 = poly2.fit_transform(X_train)
X_test_p2 = poly2.transform(X_test)
model2 = LinearRegression().fit(X_train_p2, y_train)
y_pred_quad_test = model2.predict(X_test_p2)

poly9 = PolynomialFeatures(degree=9, include_bias=False)
X_train_p9 = poly9.fit_transform(X_train)
X_test_p9 = poly9.transform(X_test)
model9 = LinearRegression().fit(X_train_p9, y_train)
y_pred_high_test = model9.predict(X_test_p9)

diagnostics = pd.DataFrame({
    "model": ["linear (underfit)", "quadratic (good)", "degree 9 (overfit)"],
    "train R2": [
        LinearRegression().fit(X_train, y_train).score(X_train, y_train),
        model2.score(X_train_p2, y_train),
        model9.score(X_train_p9, y_train),
    ],
    "test R2": [
        LinearRegression().fit(X_train, y_train).score(X_test, y_test),
        model2.score(X_test_p2, y_test),
        model9.score(X_test_p9, y_test),
    ],
    "residual pattern": ["U shape - systematic", "random cloud", "random cloud"],
    "diagnosis": ["HIGH BIAS", "correct model", "HIGH VARIANCE"],
    "fix": ["add curvature", "ship it", "regularise / more data"],
})
print(diagnostics.round(4).to_string(index=False))
print()
print("The overfitted model's residuals on the test set still LOOK random. That is")
print("the trap: residual plots must be made on TRAINING data to expose overfitting.")
print("On test data, overfitting shows up as a bad score, not an ugly plot.")


# ======================================================================================
# Visualisation
# ======================================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# --- Plot 1: the bias-variance curve ---------------------------------------------------
ax = axes[0, 0]
ax.plot(sweep_df["degree"], sweep_df["train_r2"], "o-", color="#E03131",
        linewidth=2, label="train R2")
ax.plot(sweep_df["degree"], sweep_df["test_r2"], "o-", color="#4C6EF5",
        linewidth=2, label="test R2")
ax.fill_between(sweep_df["degree"], sweep_df["train_r2"], sweep_df["test_r2"],
                color="#E03131", alpha=0.13, label="the gap = variance")
ax.axvline(best_idx + 1, color="#0CA678", linestyle="--", linewidth=2,
           label=f"sweet spot (degree {int(sweep_df.loc[best_idx, 'degree'])})")
ax.set_title("Bias-variance tradeoff as degree rises", fontsize=11, fontweight="bold")
ax.set_xlabel("polynomial degree (model complexity)")
ax.set_ylabel("R2")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Plot 2: the three fits over the data ---------------------------------------------
ax = axes[0, 1]
grid = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
ax.scatter(X_train, y_train, s=14, alpha=0.35, color="#868E96", edgecolors="none",
           label="train data")
ax.plot(grid.ravel(), true_function(grid.ravel()), color="#212529", linewidth=2.5,
        label="true function")
ax.plot(grid.ravel(), LinearRegression().fit(X_train, y_train).predict(grid),
        color="#E03131", linewidth=2, label="degree 1 (underfit)")
ax.plot(grid.ravel(), model2.predict(poly2.transform(grid)), color="#0CA678",
        linewidth=2, label="degree 2 (correct)")
ax.plot(grid.ravel(), model9.predict(poly9.transform(grid)), color="#7048E8",
        linewidth=1.6, alpha=0.85, label="degree 9 (overfit)")
ax.set_title("Three fits, one dataset", fontsize=11, fontweight="bold")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Plot 3: residual shapes, all three on TRAIN data ---------------------------------
ax = axes[1, 0]
models_and_preds = [
    ("degree 1", y_train - LinearRegression().fit(X_train, y_train).predict(X_train), "#E03131"),
    ("degree 2", y_train - model2.predict(X_train_p2), "#0CA678"),
    ("degree 9", y_train - model9.predict(X_train_p9), "#7048E8"),
]
for label, residuals, color in models_and_preds:
    ax.scatter(y_train, residuals, s=10, alpha=0.3, color=color, label=label, edgecolors="none")
ax.axhline(0, color="#212529", linewidth=1.5, linestyle="--")
ax.set_title("TRAIN residuals: U shape = bias, wide = variance", fontsize=11, fontweight="bold")
ax.set_xlabel("fitted value")
ax.set_ylabel("residual")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Plot 4: train vs test error curves, averaged over splits -------------------------
ax = axes[1, 1]
ax.errorbar(avg_df["degree"], avg_df["train_mean"], yerr=avg_df["train_std"],
            marker="o", color="#E03131", linewidth=2, capsize=4, label="train R2 (+/- 1 sd)")
ax.errorbar(avg_df["degree"], avg_df["test_mean"], yerr=avg_df["test_std"],
            marker="o", color="#4C6EF5", linewidth=2, capsize=4, label="test R2 (+/- 1 sd)")
ax.axhline(1 - NOISE_STD**2 / np.var(y), color="#868E96", linestyle="--", linewidth=2,
           label="noise ceiling")
ax.set_title("Mean +/- std over 25 random splits", fontsize=11, fontweight="bold")
ax.set_xlabel("polynomial degree")
ax.set_ylabel("R2")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

fig.suptitle("06 - Bias, Variance and Diagnosing Overfitting", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
print("1. Test Error = Bias^2 + Variance + Noise. Know your noise ceiling first.")
print("2. Underfitting (HIGH BIAS): train AND test scores are both poor. The model")
print("   class cannot represent the truth -> add flexibility, never just iterate more.")
print("3. Overfitting (HIGH VARIANCE): train score is high, test score is poor or")
print("   negative -> regularise, select features, or collect more data.")
print("4. The train-test gap is your variance estimate. Track it, not just the score.")
print("5. Residual patterns must be inspected on TRAINING data to expose overfitting.")
print("6. Selecting a degree on a single split is selection on noise. Average over splits.")
print("7. Lower training error is not progress. Cross-validation is the only honest judge.")
