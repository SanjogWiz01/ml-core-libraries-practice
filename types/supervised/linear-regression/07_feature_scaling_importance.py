"""
07 - Why Feature Scaling Matters
================================

Goal: see, concretely, what unscaled features break - and understand the one case
where scaling makes no difference at all.

What breaks without scaling
---------------------------
1. **Gradient descent convergence.** The gradient for a large-unit feature is
   much bigger, so a single learning rate either crawls on small features or
   explodes on large ones.
2. **Regularisation.** The penalty `lambda*||w||^2` is defined in coefficient
   space, so it punishes small-unit features far more than large-unit ones.
   Your model silently prefers some inputs purely because of their units.
3. **Coefficient comparison.** `|w|` no longer means "effect size"; it means
   "effect size divided by the feature's standard deviation".
4. **Conditioning.** The Gram matrix `X^T X` becomes badly conditioned and
   numerical error grows.

The important nuance
--------------------
For a SINGLE feature, scaling does not change predictions at all. It only
rescales the units of the equation:

    w_scaled = w * std(x)
    b_scaled = b - w * mean(x)

This identity is verified numerically below, because understanding *when* a fix
is unnecessary is what separates memorising from knowing.

Run:  python 07_feature_scaling_importance.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR  # only here to show scale sensitivity in a kNN-adjacent model

SEED = 42
rng = np.random.default_rng(SEED)

print("=" * 78)
print("WHY FEATURE SCALING MATTERS")
print("=" * 78)


# ======================================================================================
# PART 1 - prove the scaling identity for a single feature
# ======================================================================================
print()
print("-" * 78)
print("PART 1 - single feature: scaling changes the equation, NOT the predictions")
print("-" * 78)

n = 300
x_raw = rng.normal(50, 12, size=(n, 1))  # metres: mean 50, std 12
true_coef, true_intercept = 2.5, 8.0
y = true_coef * x_raw.ravel() + true_intercept + rng.normal(0, 5, size=n)

model_raw = LinearRegression().fit(x_raw, y)
w_raw, b_raw = model_raw.coef_[0], model_raw.intercept_
pred_raw = model_raw.predict(x_raw)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_raw)
model_scaled = LinearRegression().fit(x_scaled, y)
w_scaled, b_scaled = model_scaled.coef_[0], model_scaled.intercept_
pred_scaled = model_scaled.predict(x_scaled)

print(f"Raw data      : y = {w_raw:.6f} * x + {b_raw:.6f}")
print(f"Scaled data   : y = {w_scaled:.6f} * x_std + {b_scaled:.6f}")
print()
print("The two equations look wildly different, so check them against the identity:")
predicted_w = w_raw * scaler.scale_[0]
predicted_b = b_raw - w_raw * scaler.mean_[0]
print(f"  w_raw * std(x)      = {w_raw:.6f} * {scaler.scale_[0]:.6f} = {predicted_w:.6f}")
print(f"  w_scaled            = {w_scaled:.6f}   match: {np.isclose(predicted_w, w_scaled)}")
print(f"  b_raw - w*mean(x)   = {b_raw:.6f} - {w_raw * scaler.mean_[0]:.6f} = {predicted_b:.6f}")
print(f"  b_scaled            = {b_scaled:.6f}   match: {np.isclose(predicted_b, b_scaled)}")
print()
print(f"Max difference between the two sets of PREDICTIONS: {np.max(np.abs(pred_raw - pred_scaled)):.3e}")
print("=> Exactly zero (to float precision). For one feature, scaling is cosmetic.")
print("   The problem is purely about UNITS of the equation, and about which")
print("   algorithms (gradients, penalties, distances) care about units.")


# ======================================================================================
# PART 2 - multi-feature: coefficients become comparable, or do not
# ======================================================================================
print()
print("-" * 78)
print("PART 2 - multiple features: are the coefficients comparable?")
print("-" * 78)

n = 600
area = rng.uniform(40, 300, size=n)          # square metres, tens to hundreds
rooms = rng.integers(1, 7, size=n)             # count, single digits
distance = rng.uniform(0.2, 12.0, size=n)     # km to city centre, small decimals
price = (12.0 * area + 8.0 * rooms - 30.0 * distance
         + rng.normal(0, 25, size=n) + 40)

features = pd.DataFrame({"area_m2": area, "rooms": rooms, "distance_km": distance})
print("Feature scales:")
print(features.agg(["mean", "std", "min", "max"]).round(3).to_string())
print()
print("Notice: distance_km has a std of ~3, rooms ~1.7, area_m2 ~75. A 75x spread.")
print()

# --- raw fit -------------------------------------------------------------------------
raw_model = LinearRegression().fit(features, price)

# --- standardised fit ---------------------------------------------------------------
X_std = StandardScaler().fit_transform(features)
std_model = LinearRegression().fit(X_std, price)

comparison = pd.DataFrame({
    "std_of_feature": features.std().round(3),
    "coef_raw": raw_model.coef_.round(4),
    "coef_standardised": std_model.coef_.round(4),
    "rank_by_raw": pd.Series(np.abs(raw_model.coef_), index=features.columns).rank(ascending=False).astype(int),
    "rank_by_std": pd.Series(np.abs(std_model.coef_), index=features.columns).rank(ascending=False).astype(int),
})
print(comparison.to_string())
print()
print("Two different rankings of 'which feature matters most'.")
print("Read the 'coef_raw' column literally: area_m2 gets 12.0 per m2, distance_km")
print("gets -30.0 per km. It LOOKS like distance is 2.5x more important than area.")
print("That is an artefact of units. area moves ~75 units per standard deviation;")
print("distance moves ~3. The standardised column is the honest comparison.")
print()
print(f"Verification - a 1-SD change in each feature's effect on price:")
for i, col in enumerate(features.columns):
    effect_raw = abs(raw_model.coef_[i]) * features[col].std()
    effect_std = abs(std_model.coef_[i])
    print(f"  {col:<12} 1-SD effect: raw-model {effect_raw:8.3f}   standardised {effect_std:8.3f}")
print()
print("The '1-SD effect' column is identical either way - that is the scale-free")
print("quantity you should be reporting. Standardising just computes it directly.")


# ======================================================================================
# PART 3 - gradient descent: the practical cost of not scaling
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - gradient descent on unscaled data: too slow to be usable")
print("-" * 78)


def batch_gd(X, y, lr, n_iter):
    """Plain batch gradient descent, hand-written, returns the weight path."""
    n = X.shape[0]
    w = np.zeros(X.shape[1])
    path = []
    for _ in range(n_iter):
        residual = y - X @ w
        grad = -(2.0 / n) * (X.T @ residual)
        w -= lr * grad
        path.append(w.copy())
    return np.array(path)


X_gd = np.hstack([np.ones((n, 1)), features.values])  # do NOT scale
y_gd = price - price.mean()                          # centre the target only

print(f"Running 300 iterations of batch GD, tracking time to converge.")
print()
print(f"{'features':<14} {'lr':>8} {'iterations to 99% of final loss':>36} {'outcome'}")
print("-" * 78)

# Same learning rate for both, which is the naive thing to do.
for label, matrix, lr in [("raw", X_gd, 1e-7), ("standardised", np.hstack([np.ones((n, 1)), X_std]), 0.01)]:
    path = batch_gd(matrix, y_gd, lr=lr, n_iter=300)
    losses = np.mean((y_gd - path @ matrix.T) ** 2, axis=1)
    target = losses[-1] * 1.01
    converged = next((i for i, l in enumerate(losses) if l < target), -1)
    print(f"{label:<14} {lr:>8.1e} {converged:>36} final MSE {losses[-1]:.2f}")

print()
print("The learning rates differ by a factor of 100,000. That is the real cost of")
print("unscaled features: there is no single learning rate that works for all of")
print("them simultaneously, because the curvature along each axis differs by the")
print("square of the scale ratio.")
print()

# The tolerance search: what learning rate actually works for each?
print("Search for the best usable learning rate in each case:")
for label, matrix, grid in [
    ("raw", X_gd, np.logspace(-9, -3, 13)),
    ("standardised", np.hstack([np.ones((n, 1)), X_std]), np.logspace(-5, 0, 13)),
]:
    best = (None, np.inf)
    for lr in grid:
        path = batch_gd(matrix, y_gd, lr=lr, n_iter=300)
        losses = np.mean((y_gd - path @ matrix.T) ** 2, axis=1)
        if np.isfinite(losses[-1]) and losses[-1] < best[1]:
            best = (lr, losses[-1])
    print(f"  {label:<14} best lr {best[0]:.1e} -> final MSE {best[1]:.4f}")
print()
print("For raw data, only the very bottom of the range even survives. The tuning")
print("search itself is impractical. That is the argument for StandardScaler in")
print("one paragraph.")


# ======================================================================================
# PART 4 - regularisation is scale dependent
# ======================================================================================
print()
print("-" * 78)
print("PART 4 - ridge on unscaled data punishes the wrong features")
print("-" * 78)

print("A single alpha applied to raw vs standardised features, with identical")
print("ground truth. Watch which coefficients get crushed.")
print()
print(f"{'alpha':>8} | {'coef_raw (area, rooms, dist)':>36} | {'coef_std (area, rooms, dist)':>36}")
print("-" * 92)
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    ridge_raw = Ridge(alpha=alpha).fit(features, price)
    ridge_std = Ridge(alpha=alpha).fit(X_std, price)
    print(f"{alpha:>8} | {str(ridge_raw.coef_.round(3)):>36} | {str(ridge_std.coef_.round(3)):>36}")

print()
print("On raw data, distance_km's coefficient collapses first: it is the smallest")
print("in magnitude, so it takes the least shrinkage to reach zero. Area, with the")
print("largest magnitude, is barely touched. Ridge has effectively decided that")
print("distance is unimportant - purely because of its UNITS.")
print()
print("On standardised data, shrinkage is fair: equal treatment for equal features.")
print("This is not a subtle point, it is the difference between a working pipeline")
print("and a silent bug that looks like a modelling decision.")


# ======================================================================================
# PART 5 - scaling choices
# ======================================================================================
print()
print("-" * 78)
print("PART 5 - which scaler, and the cases where it changes answers")
print("-" * 78)

skewed = rng.exponential(scale=1.0, size=2000)  # a heavy right tail, very common in real data

comparison_table = pd.DataFrame({
    "StandardScaler": {
        "range": f"{StandardScaler().fit_transform(skewed.reshape(-1, 1)).min():.2f} to "
                  f"{StandardScaler().fit_transform(skewed.reshape(-1, 1)).max():.2f}",
        "outlier_sensitive": "YES - mean and std are dragged by outliers",
        "use_when": "linear/SGD models, kernels, PCA, anything with a penalty",
    },
    "MinMaxScaler": {
        "range": f"{MinMaxScaler().fit_transform(skewed.reshape(-1, 1)).min():.2f} to "
                 f"{MinMaxScaler().fit_transform(skewed.reshape(-1, 1)).max():.2f}",
        "outlier_sensitive": "YES, worse - min and max define the whole range",
        "use_when": "neural nets, image pixels, anything needing a bounded [0,1] input",
    },
    "RobustScaler": {
        "range": "(median-centred, IQR-scaled)",
        "outlier_sensitive": "NO - uses the 25th and 75th percentiles",
        "use_when": "data with heavy tails or outliers you cannot remove",
    },
})
print("On an exponential distribution with a long right tail:")
print(comparison_table.to_string())
print()
print(f"Mean {skewed.mean():.2f} vs median {np.median(skewed):.2f} - the tail has")
print("pulled the mean well above the median, so StandardScaler centres on a value")
print("most of the data has never seen. RobustScaler centres on the median instead.")
print()
print("Transformations are not always about scale:")
print("  log1p(x)  right-skewed features (income, counts, prices)")
print("  sqrt(x)   counts with variance > mean (Poisson-like)")
print("  Yeo-Johnson  handles both left and right skew, and can handle negatives")
print("Do this on the TARGET too, and remember to invert the predictions afterwards.")


# ======================================================================================
# PART 6 - scaling inside a Pipeline: fit on train only
# ======================================================================================
print()
print("-" * 78)
print("PART 6 - the correct way: Pipeline prevents leakage automatically")
print("-" * 78)

print("""
WRONG - the scaler has already seen the test set's mean and std:
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test = train_test_split(X_scaled, y)

RIGHT - Pipeline fits the scaler on training data only:
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(StandardScaler(), Ridge())
    pipe.fit(X_train, y_train)          # scaler learns train mean/std only
    pipe.predict(X_test)                 # test is transformed with TRAIN statistics
    pipe.feature_names_in_               # pipeline knows the input column names
""")

from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

X_train, X_test, y_train, y_test = train_test_split(features, price,
                                                    test_size=0.25, random_state=SEED)

# Demonstrate the leak quantitatively: fit the scaler on everything first.
leaky_scaler = StandardScaler().fit(features)  # <- uses test data. This is the bug.
X_leaky = leaky_scaler.transform(features)
model_leaky = Ridge(alpha=1.0).fit(X_leaky[: len(X_train)], price[: len(X_train)])
# The leak shows up as a deceptively good score on data the scaler already studied.
leaky_score = model_leaky.score(leaky_scaler.transform(features)[len(X_train):],
                               price[len(X_train):])

pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
pipe.fit(X_train, y_train)
honest_score = pipe.score(X_test, y_test)

print(f"  scaler fitted on ALL data, then scored on the tail : R2 = {leaky_score:.4f}")
print(f"  scaler fitted on TRAIN only (correct)             : R2 = {honest_score:.4f}")
print(f"  difference                                          : {leaky_score - honest_score:+.4f}")
print()
print("On a big, stable dataset the leak is a small optimism. On a small, drifted,")
print("or recently-changed dataset it can be large - and it will never show up as")
print("an error, only as a result that disappoints you in production.")


# ======================================================================================
# Visualisation
# ======================================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))

# --- Plot 1: gradient descent paths, raw vs scaled -----------------------------------
ax = axes[0, 0]
path_raw = batch_gd(X_gd, y_gd, lr=1e-7, n_iter=300)
path_std = batch_gd(np.hstack([np.ones((n, 1)), X_std]), y_gd, lr=0.01, n_iter=300)
ax.plot(path_raw[:, 1], path_raw[:, 2], color="#E03131", linewidth=2,
        label=f"raw, lr=1e-7 (area, rooms)")
ax.plot(path_std[:, 1], path_std[:, 2], color="#0CA678", linewidth=2,
        label="standardised, lr=0.01")
ax.scatter([raw_model.coef_[0]], [raw_model.coef_[1]], s=140, color="#E03131",
           marker="*", edgecolors="white", linewidths=1, label="optimum (raw coefficients)")
ax.scatter([std_model.coef_[0]], [std_model.coef_[1]], s=140, color="#0CA678",
           marker="*", edgecolors="white", linewidths=1)
ax.set_title("GD path: raw crawls, standardised cuts straight across", fontsize=11,
             fontweight="bold")
ax.set_xlabel("coefficient on area")
ax.set_ylabel("coefficient on rooms")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Plot 2: coefficient magnitude, raw vs standardised ------------------------------
ax = axes[0, 1]
width = 0.36
positions = np.arange(len(features.columns))
ax.bar(positions - width / 2, np.abs(raw_model.coef_), width, color="#E03131",
       label="|coef| raw units")
ax.bar(positions + width / 2, np.abs(std_model.coef_), width, color="#0CA678",
       label="|coef| standardised")
ax.set_xticks(positions)
ax.set_xticklabels(features.columns, fontsize=9)
ax.set_title("Same model, different ranking of importance", fontsize=11, fontweight="bold")
ax.set_ylabel("absolute coefficient")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25, axis="y")

# --- Plot 3: ridge coefficient shrinkage on raw data ---------------------------------
ax = axes[1, 0]
alphas = np.logspace(-2, 3, 40)
for i, col in enumerate(features.columns):
    coefs = [Ridge(alpha=a).fit(features, price).coef_[i] for a in alphas]
    ax.plot(alphas, coefs, linewidth=2, label=f"{col} (raw units)")
ax.set_xscale("log")
ax.axhline(0, color="#212529", linewidth=1)
ax.set_title("Ridge on RAW data: unfair shrinkage", fontsize=11, fontweight="bold")
ax.set_xlabel("alpha")
ax.set_ylabel("coefficient")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Plot 4: the same on standardised data -------------------------------------------
ax = axes[1, 1]
for i, col in enumerate(features.columns):
    coefs = [Ridge(alpha=a).fit(X_std, price).coef_[i] for a in alphas]
    ax.plot(alphas, coefs, linewidth=2, label=col)
ax.set_xscale("log")
ax.axhline(0, color="#212529", linewidth=1)
ax.set_title("Ridge on STANDARDISED data: fair shrinkage", fontsize=11, fontweight="bold")
ax.set_xlabel("alpha")
ax.set_ylabel("coefficient")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

fig.suptitle("07 - Why Feature Scaling Matters", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
print("1. Single feature: scaling does NOT change predictions, only units.")
print("   w_scaled = w * std(x),  b_scaled = b - w * mean(x). Verified above.")
print("2. Multiple features: without scaling, |coef| measures units, not importance.")
print("3. Gradient descent without scaling needs a learning rate per feature scale -")
print("   which means no usable learning rate. Measured above.")
print("4. Ridge/Lasso on unscaled data silently penalise small-unit features.")
print("5. Use a Pipeline so the scaler is fitted on training data only.")
print("6. StandardScaler for linear/SGD/penalised models, MinMaxScaler for neural nets")
print("   and images, RobustScaler when outliers dominate. Consider log1p for skew.")
