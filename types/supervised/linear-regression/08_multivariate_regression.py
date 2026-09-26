"""
08 - Multivariate (Multiple-Feature) Linear Regression
======================================================

Goal: fit a model with many features and learn how to read - and when to distrust -
each individual coefficient.

The central idea
----------------
A coefficient is the effect of one feature while all others are held constant.
That is a very different statement from "this feature matters". The two come apart
whenever features are correlated, and this file shows exactly how they come apart.

Also covered
------------
- Building and inspecting a realistic multi-feature dataset
- Ranking features properly (standardised coefficients, not raw ones)
- The duplicate-feature trap: identical columns, wildly different coefficients
- Multicollinearity previewed (full treatment in script 16)
- Interaction terms: when "additive" is the wrong model
- Confidence intervals for coefficients, via the covariance matrix
- What `coef_` ordering depends on, and why sorting it means nothing without names

Run:  python 08_multivariate_regression.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

SEED = 42
rng = np.random.default_rng(SEED)

print("=" * 78)
print("MULTIVARIATE LINEAR REGRESSION")
print("=" * 78)


# ======================================================================================
# PART 1 - build a realistic dataset
# ======================================================================================
print()
print("-" * 78)
print("PART 1 - a house-price dataset with 8 features of different types")
print("-" * 78)

n = 800

area = rng.normal(180, 45, size=n)                    # sqm
bedrooms = np.clip(rng.poisson(3, size=n) + 1, 1, 7)  # count
bathrooms = np.clip(rng.poisson(1.6, size=n), 1, 4)    # count
age = np.clip(rng.exponential(18, size=n), 0, 100)     # years, right-skewed
garage = rng.binomial(1, 0.6, size=n)                 # 0 or 1
quality = rng.normal(5, 1.4, size=n)                  # 1-10 rating
school = rng.normal(8, 1.8, size=n)                    # 1-10 rating
# Distance is partly determined by area and age - urban plots are small and new.
# That shared dependence is exactly what creates collinearity later.
distance = 0.012 * area + 0.35 * age + rng.normal(6, 2.2, size=n)
distance = np.clip(distance, 0.2, None)

house = pd.DataFrame({
    "area_sqm": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "age_years": age,
    "garage": garage,
    "build_quality": quality,
    "school_rating": school,
    "distance_km": distance,
})

# The ground truth we will try to recover.
TRUE = {
    "area_sqm": 320.0,
    "bedrooms": 18000.0,
    "bathrooms": 22000.0,
    "age_years": -900.0,
    "garage": 15000.0,
    "build_quality": 22000.0,
    "school_rating": 18000.0,
    "distance_km": -8000.0,
}
price = sum(TRUE[c] * house[c] for c in house.columns) + 45000 + rng.normal(0, 18000, size=n)
house["price"] = price

print(f"Dataset: {house.shape[0]} houses x {house.shape[1]} columns (7 features + target)")
print(house.head(5).round(1).to_string(index=False))
print()
print("Notice: distance_km is partly a function of area_sqm and age_years. Those")
print("features are correlated, and that will matter in a moment.")
print()
print(house[house.columns[:-1]].corr(numeric_only=True)["area_sqm"].round(3).to_string())
print("  <- area_sqm and distance_km are clearly correlated. Keep this in mind.")

X = house.drop(columns="price")
y = house["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)


# ======================================================================================
# PART 2 - fit and read the coefficients
# ======================================================================================
print()
print("-" * 78)
print("PART 2 - fit, then read the coefficients properly")
print("-" * 78)

model = LinearRegression().fit(X_train, y_train)
print(f"Intercept: {model.intercept_:,.0f} currency units")
print(f"Test R2  : {model.score(X_test, y_test):.4f}")
print(f"Test RMSE: {np.sqrt(np.mean((y_test - model.predict(X_test)) ** 2)):,.0f}")
print()

raw_table = pd.DataFrame({
    "feature": X.columns,
    "coef_raw": model.coef_,
    "abs": np.abs(model.coef_),
    "true_coef": [TRUE[c] for c in X.columns],
}).sort_values("abs", ascending=False).reset_index(drop=True)
raw_table["correct_sign"] = np.sign(raw_table["coef_raw"]) == np.sign(raw_table["true_coef"])
print("Coefficients in RAW units, sorted by magnitude:")
print(raw_table.round(0).to_string(index=False))
print()
print("PROBLEM: bedrooms has a coefficient of ~18,000 and distance_km ~-8,000, but")
print("their raw 'importance' is not comparable. The features have different units")
print("and very different spreads. Sorting by |coef| here is meaningless.")
print()

# Standardised coefficients: comparable, and the correct way to rank features.
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
model_std = LinearRegression().fit(X_train_std, y_train)
y_pred_std = model_std.fit(X_train_std, y_train).predict(X_train_std)

std_coefs = (X_train_std * (y_train - y_train.mean()).values[:, None]).mean(axis=0) / X_train_std.var(axis=0)
# The explicit formula above is the textbook "correlation times y-std" version.
# scikit-learn's standardised coef is simply coef_ / feature_std, which is cleaner:
std_coefs_direct = model.coef_ / X_train.std().values

std_table = pd.DataFrame({
    "feature": X.columns,
    "coef_standardised": std_coefs_direct,
    "effect_per_1sd": std_coefs_direct * y_train.std(),
    "true_effect_per_1sd": [TRUE[c] * X_train[c].std() for c in X.columns],
}).sort_values("effect_per_1sd", key=np.abs, ascending=False).reset_index(drop=True)

print("Standardised coefficients (comparable across features):")
print(std_table.round(1).to_string(index=False))
print()
print("'effect_per_1sd' answers the question you actually care about: how much does")
print("the prediction change when this feature moves by one standard deviation?")
print("The standardised coefficient IS that number, divided by the target's std.")
print()
print("These rankings are the ones to put in a report.")


# ======================================================================================
# PART 3 - the duplicate-feature trap
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - add an EXACT COPY of a feature and watch the coefficients shatter")
print("-" * 78)

X_dup = X_train.copy()
X_dup["area_sqm_COPY"] = X_train["area_sqm"]  # literally the same numbers
dup_model = LinearRegression().fit(X_dup, y_train)
dup_table = pd.DataFrame({
    "feature": X_dup.columns,
    "coef_with_duplicate": dup_model.coef_,
})
# Run the same fit many times with different splits to show the coefficients SWING.
runs = []
for split_seed in range(12):
    X_tr, X_te, y_tr, y_te = train_test_split(X_dup, y_train, test_size=0.25,
                                              random_state=split_seed)
    m = LinearRegression().fit(X_tr, y_tr)
    runs.append(m.coef_)
runs = np.array(runs)

instability = pd.DataFrame({
    "feature": X_dup.columns,
    "mean_coef": runs.mean(axis=0),
    "std_across_12_fits": runs.std(axis=0),
    "min": runs.min(axis=0),
    "max": runs.max(axis=0),
})
print("Coefficients of one dataset refit on 12 different 75/25 splits:")
print(instability.round(0).to_string(index=False))
print()
print("The two area columns swing wildly - sometimes +30,000 each, sometimes")
print("near zero, sometimes one positive and the other negative. The SUM stays")
print("roughly constant, because the model only needs their combined effect.")
print()
print("Yet the PREDICTIONS are identical in quality:")
print(f"  R2 with the original features    : {model.score(X_test, y_test):.6f}")
dup_test = X_test.copy()
dup_test["area_sqm_COPY"] = X_test["area_sqm"]
print(f"  R2 with the duplicated feature   : {dup_model.score(dup_test, y_test):.6f}")
print()
print("That is multicollinearity in its purest form:")
print("  - as a PREDICTOR: the extra column is useless, it changes nothing")
print("  - as an EXPLANATION: coefficients become arbitrary and uninterpretable")
print("  - as a DIRECTORY: |correlation| and VIF tell you which columns are the problem")


# ======================================================================================
# PART 4 - interaction terms
# ======================================================================================
print()
print("-" * 78)
print("PART 4 - interaction terms: when additive is the wrong model")
print("-" * 78)

# Synthetic case: effect of quality depends on area. A big house is worth more per
# point of quality than a small one. No additive model can express that.
m = 600
size = rng.uniform(50, 250, size=m)
qual = rng.normal(5, 1.5, size=m)
target = 1000 * size + 5000 * qual + 25 * size * qual + rng.normal(0, 800, size=m)
# ^ the 25*size*qual term is the interaction, and it is the DOMINANT effect.

X_int = pd.DataFrame({"size": size, "quality": qual})

add_model = LinearRegression().fit(X_int, target)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_int_poly = poly.fit_transform(X_int)
int_model = LinearRegression().fit(X_int_poly, target)

print(f"Terms generated: {list(poly.get_feature_names_out(['size', 'quality']))}")
print()
print("Additive model (assumes effects add independently):")
print(f"  coefficients  size {add_model.coef_[0]:8.1f}   quality {add_model.coef_[1]:8.1f}")
print(f"  R2            {add_model.score(X_int, target):.4f}")
print()
print("With the interaction term included:")
for name, coef in zip(poly.get_feature_names_out(["size", "quality"]), int_model.coef_):
    print(f"  {name:<16} {coef:10.2f}")
print(f"  R2            {int_model.score(X_int_poly, target):.4f}")
print()
print("The interaction term is the largest coefficient by far, and the additive")
print("model misses it entirely. Practical consequence: a model that assumes")
print("'each feature has one fixed effect' will be systematically wrong in the")
print("corners of your feature space - exactly where the important cases often are.")
print()
print("How to include interactions without exploding the feature count:")
print("  - only the ones theory suggests (usually a handful)")
print("  - PolynomialFeatures(degree=2) on a SMALL subset of numeric columns")
print("  - tree models learn interactions automatically - often the better answer")


# ======================================================================================
# PART 5 - confidence intervals for coefficients
# ======================================================================================
print()
print("-" * 78)
print("PART 5 - how uncertain is each coefficient?")
print("-" * 78)

# The OLS covariance matrix: Var(beta_hat) = sigma^2 * (X^T X)^-1
# Standard errors come straight out of it, and a t-interval needs a normal-ish
# residual assumption (see THEORY-GUIDE assumption 5).
X_d = X_train.values
n_obs, n_feat = X_d.shape
design = np.hstack([np.ones((n_obs, 1)), X_d])
residuals = y_train.values - design @ (np.linalg.pinv(design) @ y_train.values)
dof = n_obs - design.shape[1]
sigma_sq = np.sum(residuals**2) / dof
cov_beta = sigma_sq * np.linalg.pinv(design.T @ design)
std_errors = np.sqrt(np.diag(cov_beta))

from scipy import stats  # noqa: E402

t_critical = stats.t.ppf(0.975, dof)
ci = pd.DataFrame({
    "feature": ["intercept"] + list(X.columns),
    "coef": np.concatenate([[model.intercept_], model.coef_]),
    "std_error": std_errors,
    "t_stat": np.concatenate([[np.nan], model.coef_ / std_errors[1:]]),
})
ci["ci_low"] = ci["coef"] - t_critical * ci["std_error"]
ci["ci_high"] = ci["coef"] + t_critical * ci["std_error"]
ci["true_coef"] = [np.nan] + [TRUE[c] for c in X.columns]
ci["significant"] = np.where(ci["ci_low"] * ci["ci_high"] > 0, "yes", "NO (interval crosses 0)")

print(ci.round(0).to_string(index=False))
print()
print("A 95% interval that crosses zero means: the data cannot rule out that this")
print("feature has no effect at all. That is a much more useful sentence than a")
print("p-value, and it is the honest way to talk about a coefficient.")
print()
noisy = ci[ci["significant"] != "yes"]
if len(noisy):
    print("Features whose interval crosses zero (weak or absent evidence):")
    for _, row in noisy.iterrows():
        print(f"  {row['feature']:<16} [{row['ci_low']:>10,.0f}, {row['ci_high']:>10,.0f}]")
print()
print("IMPORTANT CAVEAT: these intervals assume independent, normally distributed")
print("residuals. Both assumptions are violated by the area/distance collinearity")
print("in this dataset, so the true intervals are WIDER than shown. Interval")
print("estimates are the part of linear regression that breaks first.")


# ======================================================================================
# PART 6 - practical checklist
# ======================================================================================
print()
print("-" * 78)
print("PART 6 - how many features is too many?")
print("-" * 78)

print(f"{'n_features':>12} {'cv R2 (5-fold)':>16} {'R2 - Adjusted R2':>20} {'params/data ratio':>20}")
print("-" * 78)

# Add pure noise columns one at a time and watch cross-validation degrade.
n_noise = 20
X_noisy = np.hstack([X_train.values, rng.normal(0, 1, size=(len(X_train), n_noise))])
noise_names = [f"noise_{i}" for i in range(n_noise)]

for k in [3, 5, 7, 10, 15, 20, 27]:
    subset = X_noisy[:, :k]
    scores = cross_val_score(LinearRegression(), subset, y_train, cv=5, scoring="r2")
    m_k = LinearRegression().fit(subset, y_train)
    r2_in = m_k.score(subset, y_train)
    adj = 1 - (1 - r2_in) * (len(y_train) - 1) / (len(y_train) - k - 1)
    print(f"{k:>12} {scores.mean():>16.4f} {r2_in - adj:>20.4f} {k / len(y_train):>20.4f}")

print()
print("Cross-validated R2 is flat then declines, while the in-sample optimism")
print("(train R2 minus adjusted R2) grows monotonically. Adjusted R2 is a")
print("sanity check, not a decision procedure. Cross-validation decides.")


# ======================================================================================
# Visualisation
# ======================================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))

# --- Plot 1: coefficient comparison ---------------------------------------------------
ax = axes[0, 0]
positions = np.arange(len(X.columns))
ax.bar(positions - 0.2, raw_table.set_index("feature").loc[X.columns, "coef_raw"] / 1000,
       0.4, color="#E03131", label="raw coefficient (thousands)")
ax.bar(positions + 0.2, std_table.set_index("feature").loc[X.columns, "effect_per_1sd"] / 1000,
       0.4, color="#0CA678", label="effect per 1 SD (thousands)")
ax.set_xticks(positions)
ax.set_xticklabels(X.columns, rotation=30, ha="right", fontsize=8)
ax.set_title("Raw coefficients mislead; 1-SD effects do not", fontsize=11, fontweight="bold")
ax.set_ylabel("thousands of currency units")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25, axis="y")

# --- Plot 2: coefficient instability with the duplicate -------------------------------
ax = axes[0, 1]
for j, col in enumerate(X_dup.columns):
    if col in ("area_sqm", "area_sqm_COPY"):
        ax.plot(runs[:, j], "o-", markersize=4, linewidth=1.5,
                color="#E03131" if col == "area_sqm" else "#7048E8", label=col)
ax.axhline(TRUE["area_sqm"], color="#212529", linestyle="--", linewidth=2, label="true coefficient")
ax.set_title("Duplicate features: coefficients swing, sum stays fixed", fontsize=11,
             fontweight="bold")
ax.set_xlabel("refit on a different random split")
ax.set_ylabel("coefficient on area")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Plot 3: partial dependence for the top features ----------------------------------
ax = axes[1, 0]
grid = {}
for col in ["area_sqm", "age_years", "build_quality"]:
    lo, hi = X_train[col].min(), X_train[col].max()
    probe = X_train.median().to_dict()  # hold everything else at the median
    axis_values = np.linspace(lo, hi, 50)
    rows = pd.DataFrame([probe] * 50)
    rows[col] = axis_values
    grid[col] = (axis_values, model.predict(rows[X.columns]))

for col, (vals, preds) in grid.items():
    ax.plot(vals, preds, linewidth=2.4, label=col)
ax.set_title("Partial dependence (other features held at median)", fontsize=11,
             fontweight="bold")
ax.set_xlabel("feature value")
ax.set_ylabel("predicted price")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Plot 4: CV score vs feature count -----------------------------------------------
k_values = [3, 5, 7, 10, 15, 20, 27]
cv_scores = [cross_val_score(LinearRegression(), X_noisy[:, :k], y_train, cv=5,
                            scoring="r2").mean() for k in k_values]
ax.plot(k_values, cv_scores, "o-", color="#4C6EF5", linewidth=2, label="5-fold CV R2")
ax.axvline(7, color="#0CA678", linestyle="--", linewidth=2, label="7 real features")
ax.set_title("Adding noise features eventually hurts", fontsize=11, fontweight="bold")
ax.set_xlabel("number of features fed to the model")
ax.set_ylabel("cross-validated R2")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

fig.suptitle("08 - Multivariate Linear Regression", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
print("1. coef_ is a bare array. Always zip it with feature names - or you have")
print("   learned nothing.")
print("2. Rank features by standardised coefficient or 1-SD effect, never by raw |coef|.")
print("3. A coefficient is a CONDITIONAL effect: 'all other features held fixed'.")
print("4. Duplicate or correlated features destroy coefficient meaning while leaving")
print("   predictions untouched. Predictions safe, explanations worthless.")
print("5. Include interaction terms when effects genuinely depend on each other;")
print("   PolynomialFeatures(degree=2) on a few columns, or let a tree find them.")
print("6. Report 95% confidence intervals, not point estimates. An interval crossing")
print("   zero is the honest answer for a weak feature.")
print("7. Cross-validation, not training R2 or adjusted R2, decides how many features.")
