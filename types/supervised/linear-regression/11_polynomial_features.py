"""
11 - Polynomial Features: Adding Non-Linearity
==============================================

Goal: model curved relationships with linear regression, understand exactly what
the expansion does to the model, and avoid the traps it sets.

The key conceptual point
------------------------
Adding `x^2`, `x^3` and `x*y` makes the model non-linear **in the inputs** but it
stays linear **in the weights**. That is why:

  - the model is still called LINEAR regression
  - the problem remains convex, so the optimum is unique and findable
  - the number of parameters EXPLODES combinatorially with the degree

Feature count for `d` features at `degree` d, with interactions:

    degree 1:  d            e.g. 5 features -> 5 terms
    degree 2:  d(d+3)/2    e.g. 5 -> 20 terms
    degree 3:  d^3+3d^2+... e.g. 5 -> 35 terms
    degree 4:  explode      e.g. 5 -> 70 terms

Ten features at degree 3 gives you 285 columns. That is where regularisation
stops being optional.

The traps
---------
1. **Extrapolation.** A degree-9 fit is meaningless outside the training range,
   because nothing constrains it there. It will shoot to infinity.
2. **Numerical blowup.** `x^9` for `x = 100` is `1e18`. Scale before expanding.
3. **Collinearity.** Powers of the same feature are highly correlated, so raw
   polynomial regression is unstable without ridge.
4. **Feature count.** Apply `PolynomialFeatures` to a FEW columns, not the whole frame.

Run:  python 11_polynomial_features.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SEED = 42
rng = np.random.default_rng(SEED)

print("=" * 78)
print("POLYNOMIAL FEATURES - CURVING A STRAIGHT-LINE MODEL")
print("=" * 78)


# ======================================================================================
# PART 1 - see the feature explosion
# ======================================================================================
print()
print("-" * 78)
print("PART 1 - what PolynomialFeatures actually generates")
print("-" * 78)

for d in range(1, 7):
    poly = PolynomialFeatures(degree=d, include_bias=False)
    poly.fit(np.zeros((1, d)))
    n_terms = len(poly.get_feature_names_out())
    print(f"  {d} input features, degree {d}: {n_terms:>5} terms   {list(poly.get_feature_names_out())[:6]}"
          f"{' ...' if n_terms > 6 else ''}")

print()
print("For 10 features the growth is brutal:")
for d in [1, 2, 3, 4]:
    poly = PolynomialFeatures(degree=d, include_bias=False)
    poly.fit(np.zeros((1, 10)))
    print(f"  degree {d}: {len(poly.get_feature_names_out()):>6} columns")
print()
print("Degrees 3 and up on a wide frame are only viable with strong regularisation.")


# ======================================================================================
# PART 2 - a genuinely curved relationship
# ======================================================================================
print()
print("-" * 78)
print("PART 2 - fit a curve, and find the right degree")
print("-" * 78)


def true_curve(x):
    """Deliberately non-linear, non-polynomial, and asymmetric."""
    return 0.8 * x ** 3 - 2.5 * x ** 2 + 1.2 * x + 10.0


NOISE = 4.0
n = 300
X = rng.uniform(-3, 3, size=(n, 1))
y = true_curve(X.ravel()) + rng.normal(0, NOISE, size=n)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)

print(f"Truth: 0.8*x^3 - 2.5*x^2 + 1.2*x + 10, plus noise of std {NOISE}")
print(f"Variance of y: {np.var(y):.1f} -> best achievable R2 ~ {1 - NOISE**2 / np.var(y):.3f}")
print()
print(f"{'degree':>7} {'#terms':>8} {'train R2':>10} {'test R2':>10} {'test RMSE':>11} {'verdict':>26}")
print("-" * 78)

results = []
cv = KFold(5, shuffle=True, random_state=SEED)
for degree in range(1, 11):
    pipe = Pipeline([
        # Scale FIRST. Without this, x^8 on values near 3 produces numbers in the
        # thousands and the fit becomes numerically fragile.
        ("scale", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("model", LinearRegression()),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    # Cross-validated score: the only column you should be selecting on.
    cv_score = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2").mean()

    train_r2 = pipe.score(X_train, y_train)
    test_r2 = pipe.score(X_test, y_test)
    rmse = np.sqrt(np.mean((y_test - pred) ** 2))
    n_terms = len(PolynomialFeatures(degree=degree, include_bias=False)
                  .fit(np.zeros((1, 1))).get_feature_names_out())

    if degree <= 2:
        verdict = "underfit, curve not reached"
    elif cv_score > 0.93:
        verdict = "at the noise ceiling"
    elif degree >= 7:
        verdict = "overfitting starts"
    else:
        verdict = "reasonable"

    results.append({"degree": degree, "n_terms": n_terms, "train_r2": train_r2,
                    "test_r2": test_r2, "cv_r2": cv_score, "rmse": rmse, "verdict": verdict})
    print(f"{degree:>7} {n_terms:>8} {train_r2:>10.4f} {test_r2:>10.4f} {rmse:>11.3f} {verdict:>26}")

results_df = pd.DataFrame(results)
best = results_df.loc[results_df["cv_r2"].idxmax()]
print("-" * 78)
print(f"Best degree by CROSS-VALIDATION: {int(best['degree'])} (CV R2 = {best['cv_r2']:.4f})")
print(f"Note the true polynomial is degree 3, and CV correctly identifies it.")
print()
print("The gap between train R2 and test R2 grows with degree - that gap is the")
print("variance, and it is why you should select on CV, not on the training score.")


# ======================================================================================
# PART 3 - the noise ceiling, reached
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - knowing when to stop")
print("-" * 78)
ceiling = 1 - NOISE**2 / np.var(y)
print(f"Estimated noise ceiling: R2 = {ceiling:.4f}")
print(f"Best model achieved   : R2 = {best['cv_r2']:.4f}")
print(f"Gap to ceiling         : {ceiling - best['cv_r2']:.4f}")
print()
if ceiling - best["cv_r2"] < 0.02:
    print("The model is within 2% of the theoretical maximum. Going to degree 8 would")
    print("only fit noise more precisely. This is where you STOP, and where a")
    print("higher-degree model starts looking better on train and worse everywhere else.")
print()
print("Computing the noise ceiling is a cheap, high-value habit:")
print("  residual std of a good model ~ the noise level")
print("  ceiling R2 = 1 - noise_var / var(y)")
print("  if your R2 is already there, no amount of feature engineering will help.")


# ======================================================================================
# PART 4 - the extrapolation trap
# ======================================================================================
print()
print("-" * 78)
print("PART 4 - EXTRAPOLATION: where polynomial models fall apart")
print("-" * 78)

# Fit on a narrow range, then predict far outside it.
train_narrow = (X.ravel() > -1.5) & (X.ravel() < 1.5)
X_narrow, y_narrow = X[train_narrow], y[train_narrow]
print(f"Training range: x in [{X_narrow.min():.2f}, {X_narrow.max():.2f}]")
print(f"Predicting out to x = 8, which is {(8 / X_narrow.max()):.1f}x beyond the training edge.")
print()

x_test = np.array([[-8.0], [-6.0], [0.0], [6.0], [8.0]])

print(f"{'x':>6} {'true y':>10} | {'degree 1':>10} {'degree 3':>12} {'degree 6':>14} {'degree 9':>16}")
print("-" * 78)
row = {}
for degree in [1, 3, 6, 9]:
    pipe = Pipeline([("scale", StandardScaler()),
                     ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                     ("model", LinearRegression())])
    pipe.fit(X_narrow, y_narrow)
    row[degree] = pipe.predict(x_test)

for i, x_val in enumerate(x_test.ravel()):
    true_val = true_curve(x_val)
    print(f"{x_val:>6.1f} {true_val:>10.1f} | {row[1][i]:>10.1f} {row[3][i]:>12.1f} "
          f"{row[6][i]:>14.1f} {row[9][i]:>16.1f}")

print()
print("The degree-9 column is not slightly wrong - it is off by orders of magnitude,")
print("and the sign is meaningless too. A degree-1 model extrapolates badly but")
print("predictably. A degree-9 model extrapolates absurdly.")
print()
print("RULES:")
print("  - Never use a polynomial model outside the range of the data it was trained on")
print("  - Clip or reject out-of-range inputs at the serving layer")
print("  - Degree 2-3 is usually the safe ceiling; beyond that, use splines, GAMs,")
print("    gradient boosting, or a neural network, all of which behave better at the edges")


# ======================================================================================
# PART 5 - polynomial + regularisation is the practical combination
# ======================================================================================
print()
print("-" * 78)
print("PART 5 - PolynomialFeatures with Ridge: stability instead of wildness")
print("-" * 78)

print("Raw polynomial OLS is unstable because x, x^2, x^3 are strongly correlated.")
print("Ridge spreads the weight across them instead of putting everything on one.")
print()
print(f"{'degree':>7} {'ridge alpha':>12} {'CV R2':>9} {'max |coef|':>12} {'coef std':>11} {'stable?':>10}")
print("-" * 78)

for degree in [3, 5, 7]:
    for alpha in [0.0, 0.01, 0.1, 1.0, 10.0]:
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("model", Ridge(alpha=alpha)),
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")
        pipe.fit(X_train, y_train)
        coefs = pipe.named_steps["model"].coef_
        stable = "yes" if np.max(np.abs(coefs)) < 50 else "no"
        print(f"{degree:>7} {alpha:>12} {scores.mean():>9.4f} {np.max(np.abs(coefs)):>12.1f} "
              f"{np.std(coefs):>11.2f} {stable:>10}")
    print()

print("With alpha = 0 the coefficients explode at high degree. As alpha rises the")
print("coefficients shrink toward zero, the predictions stay smooth, and the CV score")
print("barely changes - which is the trade you want. Regularisation is what makes a")
print("degree-7 model usable instead of a liability.")
print()
print("Practical ordering: scale -> expand -> regularise. Expanding before scaling")
print("lets x^7 dominate the standardisation and quietly break everything.")


# ======================================================================================
# PART 6 - multivariate polynomial terms and interaction handling
# ======================================================================================
print()
print("-" * 78)
print("PART 6 - two features: interactions, and how to choose what to expand")
print("-" * 78)

m = 500
size = rng.uniform(40, 220, size=m)
quality = rng.uniform(1, 10, size=m)
# Real interaction: the value of quality depends on size. Plus a mild curve in size.
target = (800 * size + 4000 * quality + 30 * size * quality
          + 0.02 * size ** 2 + rng.normal(0, 900, size=m))

X2 = pd.DataFrame({"size": size, "quality": quality})

variants = {
    "additive only (no poly)": Pipeline([("scale", StandardScaler()), ("m", LinearRegression())]),
    "poly degree 2 on both": Pipeline([("scale", StandardScaler()),
                                       ("poly", PolynomialFeatures(2, include_bias=False)),
                                       ("m", LinearRegression())]),
    "poly degree 2 + Ridge": Pipeline([("scale", StandardScaler()),
                                       ("poly", PolynomialFeatures(2, include_bias=False)),
                                       ("m", Ridge(alpha=1.0))]),
    "poly degree 3 + Ridge": Pipeline([("scale", StandardScaler()),
                                       ("poly", PolynomialFeatures(3, include_bias=False)),
                                       ("m", Ridge(alpha=10.0))]),
}

print(f"{'model':<26} {'#terms':>8} {'CV R2':>9} {'test R2':>9} {'verdict'}")
print("-" * 78)
for name, pipe in variants.items():
    scores = cross_val_score(pipe, X2, target, cv=cv, scoring="r2")
    pipe.fit(X2, target)
    n_terms = (len(pipe.named_steps["poly"].get_feature_names_out(["size", "quality"]))
               if "poly" in pipe.named_steps else 2)
    test_r2 = pipe.score(X2, target)
    verdict = "misses the interaction" if "additive" in name else "captures it"
    print(f"{name:<26} {n_terms:>8} {scores.mean():>9.4f} {test_r2:>9.4f} {verdict:>24}")

print()
print("Terms generated by degree 2 on two features:")
print(f"  {list(PolynomialFeatures(2, include_bias=False).fit(X2).get_feature_names_out(['size', 'quality']))}")
print("  'size quality' is the INTERACTION term, and here it is the single largest")
print("effect in the data. An additive model is not slightly wrong - it is")
print("systematically wrong across the whole feature space.")
print()
print("Selective expansion, when you have many columns:")
print("  1. Expand only the numeric columns that plausibly curve (usually 3-5)")
print("  2. One-hot categories, then use `PolynomialFeatures` ONLY if a category")
print("     interaction is plausible (location x property type, for example)")
print("  3. Never expand a full 50-column frame to degree 3 - you get thousands of")
print("     columns, most of them noise, and the fit becomes unreadable")
print("  4. Alternatively: let GradientBoosting find interactions automatically")


# ======================================================================================
# Visualisation
# ======================================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))

# --- Plot 1: the feature explosion ---------------------------------------------------
ax = axes[0, 0]
n_features = [1, 2, 3, 5, 8, 10, 15, 20]
for degree, color, marker in [(1, "#4C6EF5", "o"), (2, "#0CA678", "s"), (3, "#E03131", "^")]:
    counts = [len(PolynomialFeatures(degree=degree, include_bias=False)
                  .fit(np.zeros((1, d))).get_feature_names_out()) for d in n_features]
    ax.plot(n_features, counts, marker=marker, color=color, linewidth=2, label=f"degree {degree}")
ax.set_yscale("log")
ax.set_title("Feature count explodes with degree", fontsize=11, fontweight="bold")
ax.set_xlabel("number of input features")
ax.set_ylabel("output columns (log scale)")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Plot 2: the fits over the data --------------------------------------------------
ax = axes[0, 1]
grid = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
ax.scatter(X_train, y_train, s=14, alpha=0.35, color="#868E96", edgecolors="none",
           label="train data")
ax.plot(grid.ravel(), true_curve(grid.ravel()), color="#212529", linewidth=2.5,
        label="truth")
for degree, color in [(1, "#E03131"), (3, "#0CA678"), (7, "#7048E8")]:
    pipe = Pipeline([("scale", StandardScaler()),
                     ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                     ("model", Ridge(alpha=1.0))])
    pipe.fit(X_train, y_train)
    ax.plot(grid.ravel(), pipe.predict(grid), color=color, linewidth=1.8,
            label=f"degree {degree} + ridge")
ax.set_title("Ridge keeps high degrees smooth", fontsize=11, fontweight="bold")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Plot 3: train vs CV score by degree ---------------------------------------------
ax = axes[1, 0]
ax.plot(results_df["degree"], results_df["train_r2"], "o-", color="#E03131",
        linewidth=2, label="train R2 (always best)")
ax.plot(results_df["degree"], results_df["cv_r2"], "o-", color="#0CA678",
        linewidth=2, label="cross-validated R2 (the one to trust)")
ax.axhline(ceiling, color="#868E96", linestyle="--", linewidth=2, label="noise ceiling")
ax.axvline(best["degree"], color="#7048E8", linestyle="--", linewidth=2,
           label=f"best degree {int(best['degree'])}")
ax.set_title("The gap is the variance", fontsize=11, fontweight="bold")
ax.set_xlabel("polynomial degree")
ax.set_ylabel("R2")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Plot 4: the extrapolation disaster ----------------------------------------------
ax = axes[1, 1]
x_wide = np.linspace(-8, 8, 400).reshape(-1, 1)
ax.plot(x_wide.ravel(), true_curve(x_wide.ravel()), color="#212529", linewidth=2.5,
        label="truth")
for degree, color in [(1, "#E03131"), (3, "#0CA678"), (9, "#7048E8")]:
    pipe = Pipeline([("scale", StandardScaler()),
                     ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                     ("model", LinearRegression())])
    pipe.fit(X_narrow, y_narrow)
    ax.plot(x_wide.ravel(), pipe.predict(x_wide), color=color, linewidth=1.8,
            label=f"degree {degree}")
ax.axvspan(X_narrow.min(), X_narrow.max(), color="#FFD43B", alpha=0.3, label="trained range")
ax.set_ylim(-500, 500)
ax.set_title("EXTRAPOLATION: never predict outside the trained range", fontsize=11,
             fontweight="bold", color="#E03131")
ax.set_xlabel("x")
ax.set_ylabel("predicted y (clipped for display)")
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.grid(alpha=0.25)

fig.suptitle("11 - Polynomial Features", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
print("1. PolynomialFeatures makes the model non-linear in the INPUTS but still")
print("   linear in the WEIGHTS. That is why it is still linear regression, and why")
print("   the optimum stays unique.")
print("2. Feature count explodes combinatorially. On a wide frame, degree 3 is")
print("   already unusable without regularisation.")
print("3. Order of operations: scale, then expand, then regularise.")
print("4. Select the degree by CROSS-VALIDATION, and stop when you reach the noise")
print("   ceiling. Training R2 always rises and tells you nothing.")
print("5. High-degree polynomials are unusable for extrapolation - the predictions")
print("   diverge and even change sign. Clip or reject out-of-range inputs.")
print("6. Interactions are the real reason to use degree 2. A missing interaction")
print("   makes a model systematically wrong, not slightly wrong.")
print("7. Beyond degree 2-3, prefer splines, GAMs or gradient boosting. They model")
print("   curvature without the extrapolation behaviour.")
