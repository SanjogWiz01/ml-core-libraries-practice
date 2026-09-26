"""
02 - Linear Regression With scikit-learn
=======================================

Goal: the production way to fit linear regression, and - more importantly - how to
read what it gives back.

What this file demonstrates
---------------------------
1. The five-line scikit-learn workflow: split, fit, predict, score.
2. Reading `coef_` and `intercept_` as a sentence in plain English.
3. Why `coef_` is an array and therefore needs to be zipped with feature names.
4. `predict` on a single row, and the shapes involved.
5. Comparing the model against `DummyRegressor`, the only honest baseline.
6. `score()` and what R2 actually means on held-out data.
7. Residual inspection, the first diagnostic you should always run.

Run:  python 02_linear_regression_sklearn.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

SEED = 42  # fix the seed everywhere so the numbers in the output are reproducible

print("=" * 74)
print("LINEAR REGRESSION WITH SCIKIT-LEARN")
print("=" * 74)


# ======================================================================================
# PART 1 - the minimal workflow on a synthetic single-feature problem
# ======================================================================================
print()
print("-" * 74)
print("PART 1 - the five-line workflow")
print("-" * 74)

rng = np.random.default_rng(SEED)
n_samples = 300
X_simple = rng.uniform(1, 11, size=(n_samples, 1))
noise = rng.normal(0, 6, size=n_samples)
y_simple = 4.2 * X_simple.ravel() - 1.5 + noise

# Step 1: SPLIT. Always first, always with a seed.
# test_size=0.2 holds out 20%. random_state makes the split reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=SEED
)
print(f"train rows: {X_train.shape[0]}   test rows: {X_test.shape[0]}")
print(f"feature matrix shape: {X_train.shape}  ->  (n_samples, n_features)")

# Step 2: FIT. This is the step that "learns". model.fit only looks at training data.
model = LinearRegression()
model.fit(X_train, y_train)
print(f"\nFitted.  intercept_ = {model.intercept_:.4f}   coef_ = {model.coef_}")

# Step 3: PREDICT. This step never changes the model.
y_pred = model.predict(X_test)
print(f"Predict returned shape {y_pred.shape} - one prediction per test row")

# Step 4: SCORE. model.score uses R2 by default for regressors.
print(f"\nmodel.score(X_test, y_test) = {model.score(X_test, y_test):.4f}   (this is R2)")

# Step 5: THE HONEST BASELINE. If we cannot beat "always predict the training
# mean", the features carry no usable linear signal.
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
print(f"DummyRegressor R2          = {dummy.score(X_test, y_test):.4f}   <- the bar to beat")


# ======================================================================================
# PART 2 - reading coef_ and intercept_ as English
# ======================================================================================
print()
print("-" * 74)
print("PART 2 - interpreting the fitted equation")
print("-" * 74)

print("The fitted model is a sentence:")
print(f"    predicted_target = {model.intercept_:.3f} + ({model.coef_[0]:.3f} * x)")
print()
print("Reading it: 'each additional unit of x raises the predicted target by "
      f"{model.coef_[0]:.2f} on average'.")
print(f"'{model.intercept_:.2f}' is the prediction at x = 0. Whether that is")
print("meaningful depends on the data. If x=0 is impossible (a year, a height),")
print("the intercept is just a mathematical anchor and should not be discussed.")
print()

# Residuals: what the model got wrong. The left column is truth, the right is
# error. A good model leaves nothing systematic in this column.
comparison = pd.DataFrame({
    "actual": y_test[:8].round(2),
    "predicted": y_pred[:8].round(2),
    "residual": (y_test - y_pred)[:8].round(2),
    "abs_error": np.abs(y_test - y_pred)[:8].round(2),
})
print("First 8 test rows - actual vs predicted vs residual:")
print(comparison.to_string(index=False))
print()
print("A large positive residual means we under-predicted. A large negative one")
print("means we over-predicted. Residual PATTERNS matter more than residual size.")


# ======================================================================================
# PART 3 - multiple features, where coef_ becomes a list
# ======================================================================================
print()
print("-" * 74)
print("PART 3 - multi-feature regression: coef_ is an ARRAY, zip it carefully")
print("-" * 74)

# load_diabetes is a real regression dataset bundled with scikit-learn:
# 442 patients, 10 physiological features, target = disease progression score.
# (The famous Boston housing dataset was removed from scikit-learn for licensing
# reasons, so diabetes is the standard real-data replacement.)
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

print(f"Dataset: {X.shape[0]} patients x {X.shape[1]} features")
print(f"Features: {list(X.columns)}")
print(f"Target  : quantitative measure of disease progression, mean {y.mean():.1f}")
print()
print("First 3 rows:")
print(X.head(3).round(3).to_string(index=False))

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED)

multi = LinearRegression()
multi.fit(X_tr, y_tr)
y_te_pred = multi.predict(X_te)

# THE critical habit: never print coef_ bare. Always pair it with the feature name,
# otherwise index 3 tells you nothing and the whole table becomes unreadable.
coefficients = pd.DataFrame({
    "feature": X.columns,
    "coefficient": multi.coef_,
    "abs_coefficient": np.abs(multi.coef_),
}).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

coefficients["direction"] = np.where(coefficients["coefficient"] > 0, "+ increases", "- decreases")

print()
print("Coefficients, sorted by strength of effect:")
print(coefficients.round(3).to_string(index=False))
print()
print(f"Intercept: {multi.intercept_:.2f}")
print()
print("CAVEAT worth internalising: in a multi-feature model, a coefficient is the")
print("effect of that feature with all OTHERS HELD FIXED. It is a conditional")
print("effect, not a standalone one. When features correlate, the individual")
print("coefficients become unstable even though the predictions stay accurate.")
print("This is multicollinearity - see script 16.")

print()
print("Metrics on the test set:")
print(f"  R2   = {r2_score(y_te, y_te_pred):.4f}   (variance explained)")
print(f"  MAE  = {mean_absolute_error(y_te, y_te_pred):.2f}   (typical error, original units)")
print(f"  RMSE = {np.sqrt(mean_squared_error(y_te, y_te_pred)):.2f}   (punishes big misses)")
print(f"  MAPE = {np.mean(np.abs((y_te - y_te_pred) / y_te)) * 100:.2f}%   (relative error)")


# ======================================================================================
# PART 4 - predict a single row, and the shape rules
# ======================================================================================
print()
print("-" * 74)
print("PART 4 - predicting one new row")
print("-" * 74)

new_patient = X.iloc[[0]]  # double brackets on purpose: preserves 2-D shape
prediction = multi.predict(new_patient)[0]

print("A single patient as a 1-D array would break predict().")
print(f"  X.iloc[0].shape    = {X.iloc[0].shape}   <- 1-D, will raise an error")
print(f"  X.iloc[[0]].shape  = {new_patient.shape}   <- 2-D, correct")
print()
print(f"Predicted disease progression for patient 0: {prediction:.2f}")
print(f"Actual value for patient 0                 : {y[0]:.2f}")
print()
print("Rule: scikit-learn's predict() always wants 2-D, shape (n_samples, n_features).")
print("Use [[...]] for one row, or np.array([row]).reshape(1, -1).")


# ======================================================================================
# PART 5 - diagnostics you should run before believing any of this
# ======================================================================================
print()
print("-" * 74)
print("PART 5 - the two plots that must look boring")
print("-" * 74)

test_residuals = y_te - y_te_pred
print("Checklist:")
print("  a) R2 above the DummyRegressor baseline?")
print("     -> model beats a constant, so the features carry signal")
print(f"     -> ours: {r2_score(y_te, y_te_pred):.4f} vs dummy {r2_score(y_te, dummy.fit(X_tr, y_tr).predict(X_te)):.4f}")
print("  b) Residuals vs fitted scatter - should be a shapeless cloud.")
print("     A visible curve (U, arc, wave) = a non-linearity we are missing.")
print("  c) Residuals should straddle zero roughly evenly.")
print(f"     -> {np.mean(test_residuals):+.4f} mean residual, so no gross bias")
print(f"     -> {np.std(test_residuals):.2f} residual std")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

# --- Plot 1: the classic fit ---------------------------------------------------------
ax = axes[0]
feature_0 = X.columns[0]
ax.scatter(X_tr[feature_0], y_tr, alpha=0.35, s=20, color="#4C6EF5",
           edgecolors="none", label="train")
line_x = np.linspace(X[feature_0].min(), X[feature_0].max(), 100)
# Partial dependence on one feature: hold others at their MEAN (or median) and
# sweep this one. This is the standard, honest way to draw "the line" for
# multi-feature regression.
others = X.drop(columns=[feature_0])
reference_row = others.mean().to_numpy()
sweep = pd.DataFrame(
    np.tile(reference_row, (len(line_x), 1)),
    columns=others.columns,
)
sweep.insert(0, feature_0, line_x)
ax.plot(line_x, multi.predict(sweep), color="#E03131", linewidth=2.5,
        label="fit (other features at mean)")
ax.set_title(f"Fit on '{feature_0}'", fontsize=11, fontweight="bold")
ax.set_xlabel(feature_0)
ax.set_ylabel("target")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Plot 2: residuals vs fitted - THE diagnostic plot --------------------------------
ax = axes[1]
ax.scatter(y_te_pred, test_residuals, alpha=0.55, s=24, color="#0CA678", edgecolors="none")
ax.axhline(0, color="#E03131", linestyle="--", linewidth=2)
# A trend line through the residuals exposes hidden structure that the eye misses.
z = np.polyfit(y_te_pred, test_residuals, 1)
trend_x = np.linspace(y_te_pred.min(), y_te_pred.max(), 50)
ax.plot(trend_x, np.polyval(z, trend_x), color="#7048E8", linewidth=2,
        label=f"trend slope {z[0]:+.4f}")
ax.set_title("Residuals vs fitted (want: shapeless cloud)", fontsize=11, fontweight="bold")
ax.set_xlabel("fitted value")
ax.set_ylabel("residual")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Plot 3: residual distribution ---------------------------------------------------
ax = axes[2]
ax.hist(test_residuals, bins=28, color="#F59F00", edgecolor="white", alpha=0.9)
ax.axvline(0, color="#E03131", linestyle="--", linewidth=2)
ax.set_title("Residual distribution (want: centred on 0)", fontsize=11, fontweight="bold")
ax.set_xlabel("residual")
ax.set_ylabel("count")
ax.grid(alpha=0.25)

fig.suptitle("02 - Linear Regression With scikit-learn", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


# ======================================================================================
# Summary
# ======================================================================================
print()
print("=" * 74)
print("SUMMARY")
print("=" * 74)
print("1. The workflow is split -> fit -> predict -> score. Never skip the split.")
print("2. coef_ is a NumPy array. Always zip it with feature names before printing.")
print("3. In multi-feature models a coefficient is a CONDITIONAL effect (others fixed).")
print("4. Beat DummyRegressor before believing any score.")
print("5. predict() needs 2-D input: [[row]] for a single sample.")
print("6. Residuals must look random. A curve in the residual plot is a real signal")
print("   that a straight line cannot capture.")
