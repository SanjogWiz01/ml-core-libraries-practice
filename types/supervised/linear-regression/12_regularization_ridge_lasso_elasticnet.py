"""
12 - Regularisation: Ridge, Lasso and Elastic Net
=================================================

Goal: control the bias-variance tradeoff explicitly with a penalty term, and
understand the geometric reason Ridge and Lasso behave differently.

The three penalties
------------------
    Ridge       MSE + alpha * ||w||_2^2      (L2, squared)
    Lasso       MSE + alpha * ||w||_1        (L1, absolute)
    ElasticNet  MSE + alpha*(l1_ratio*||w||_1 + (1-l1_ratio)*||w||_2^2)

- `alpha` (written `lambda` in the textbooks) controls penalty STRENGTH.
  alpha = 0 recovers ordinary least squares. Large alpha underfits.
- `l1_ratio` (written `a` or `mix` in some libraries) is Elastic Net only:
  1.0 = pure Lasso, 0.0 = pure Ridge, 0.5 = even blend.

The geometric reason they differ
--------------------------------
The constrained view: minimise MSE subject to ||w||_1 <= t (Lasso) or
||w||_2 <= t (Ridge). The solution is where the MSE contours first touch the
constraint region.

- L1's region is a DIAMOND with corners on the coordinate axes. The first
  contact point is very often exactly on an axis, so a weight is exactly 0.
- L2's region is a CIRCLE, which touches no axis, so weights shrink smoothly
  toward zero but never reach it.

So Lasso does feature selection; Ridge does not. That is geometry, not folklore.

Run:  python 12_regularization_ridge_lasso_elasticnet.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
rng = np.random.default_rng(SEED)

print("=" * 78)
print("RIDGE, LASSO AND ELASTIC NET")
print("=" * 78)


# ======================================================================================
# PART 1 - a dataset where overfitting is easy
# ======================================================================================
print()
print("-" * 78)
print("PART 1 - 20 features, only 3 of them real")
print("-" * 78)

n_samples, n_features = 120, 20
X = rng.normal(0, 1, size=(n_samples, n_features))
true_coef = np.zeros(n_features)
true_coef[[0, 5, 11]] = [4.0, -3.0, 2.5]  # only 3 features actually matter

y = X @ true_coef + 8.0 + rng.normal(0, 1.0, size=n_samples)

feature_names = [f"f{i:02d}" for i in range(n_features)]
X_df = pd.DataFrame(X, columns=feature_names)
print(f"{n_samples} samples, {n_features} features, ratio {n_samples / n_features:.1f}:1")
print(f"Ground truth: only f00={true_coef[0]}, f05={true_coef[5]}, f11={true_coef[11]} are non-zero")
print()
print("With 17 pure-noise features and 120 rows, unregularised OLS will happily")
print("assign a large coefficient to every single one of them.")
print()

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.3, random_state=SEED)

ols = LinearRegression_placeholder = None  # readability alias resolved below
from sklearn.linear_model import LinearRegression  # noqa: E402

ols = LinearRegression().fit(X_train, y_train)
ols_coefs = ols.coef_
print(f"Unregularised OLS: train R2 {ols.score(X_train, y_train):.4f}, "
      f"test R2 {ols.score(X_test, y_test):.4f}")
print(f"  non-zero coefficients: {np.sum(np.abs(ols_coefs) > 1e-8)} of {n_features}")
print(f"  sum |coef|: {np.abs(ols_coefs).sum():.2f}  (true sum is {np.abs(true_coef).sum():.2f})")
print()
print("Every noise feature got a non-zero weight. The model memorised the sample.")
print("This is the problem regularisation solves.")


# ======================================================================================
# PART 2 - the three penalties side by side
# ======================================================================================
print()
print("-" * 78)
print("PART 2 - the same fit with each penalty")
print("-" * 78)
print(f"{'model':<28} {'alpha':>8} {'train R2':>10} {'test R2':>9} {'#nonzero':>10} {'sum |w|':>10}")
print("-" * 78)

scorer = KFold(5, shuffle=True, random_state=SEED)

candidates = [
    ("OLS (no penalty)", LinearRegression(), 0.0),
    ("Ridge", Ridge(alpha=0.1), 0.1),
    ("Ridge", Ridge(alpha=1.0), 1.0),
    ("Ridge", Ridge(alpha=10.0), 10.0),
    ("Ridge", Ridge(alpha=100.0), 100.0),
    ("Lasso", Lasso(alpha=0.05), 0.05),
    ("Lasso", Lasso(alpha=0.2), 0.2),
    ("Lasso", Lasso(alpha=0.5), 0.5),
    ("Lasso", Lasso(alpha=1.0), 1.0),
    ("ElasticNet", ElasticNet(alpha=0.1, l1_ratio=0.5), 0.1),
    ("ElasticNet", ElasticNet(alpha=0.5, l1_ratio=0.9), 0.5),
]

results = []
for name, model, alpha in candidates:
    pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
    pipe.fit(X_train, y_train)
    fitted = pipe.named_steps["model"]
    coefs = fitted.coef_
    n_nonzero = int(np.sum(np.abs(coefs) > 1e-8))
    results.append({
        "name": name, "alpha": alpha,
        "train_r2": pipe.score(X_train, y_train),
        "test_r2": pipe.score(X_test, y_test),
        "cv_r2": cross_val_score(pipe, X_train, y_train, cv=scorer, scoring="r2").mean(),
        "n_nonzero": n_nonzero,
        "sum_abs": np.abs(coefs).sum(),
    })
    r = results[-1]
    print(f"{name:<28} {alpha:>8} {r['train_r2']:>10.4f} {r['test_r2']:>9.4f} "
          f"{n_nonzero:>10} {r['sum_abs']:>10.2f}")

res_df = pd.DataFrame(results)
print("-" * 78)
print()
print("Three things to notice:")
print()
print("1. OLS keeps ALL 20 coefficients non-zero. Ridge shrinks all of them but")
print("   keeps all 20 non-zero. Lasso drives the irrelevant ones to EXACTLY zero.")
print("   That is the feature-selection behaviour, and it is unique to L1.")
print()
print("2. Lasso's test R2 is often HIGHER than Ridge's at comparable alpha, even")
print("   though its training R2 is lower. It is doing something more efficient:")
print("   rather than spreading weight thinly across 20 features, it concentrates")
print("   it on the 3 that matter.")
print()
print("3. Every penalty trades training R2 for test R2. That is the bias-variance")
print("   tradeoff made explicit and tunable in a single number.")


# ======================================================================================
# PART 3 - the coefficient path
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - the coefficient path: the single most useful plot in regularisation")
print("-" * 78)
print("Plot every coefficient as a function of alpha. Reading it tells you:")
print("  - which features survive longest (probably the real ones)")
print("  - whether features are redundant (curves move together)")
print("  - where the model becomes useless (everything hits zero)")
print()

alphas = np.logspace(-3, 3, 120)
ridge_path = np.array([Ridge(alpha=a).fit(X, y).coef_ for a in alphas])
lasso_path = np.array([Lasso(alpha=a, max_iter=5000).fit(X, y).coef_ for a in alphas])
enet_path = np.array([ElasticNet(alpha=a, l1_ratio=0.5, max_iter=5000).fit(X, y).coef_
                      for a in alphas])

print("Number of non-zero coefficients as alpha grows:")
print(f"{'alpha':>10} {'Ridge':>10} {'Lasso':>10} {'ElasticNet':>13}")
print("-" * 48)
for a in [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0, 50.0]:
    i_r = int(np.argmin(np.abs(alphas - a)))
    i_l = int(np.argmin(np.abs(alphas - a)))
    nz_ridge = int(np.sum(np.abs(ridge_path[i_r]) > 1e-8))
    nz_lasso = int(np.sum(np.abs(lasso_path[i_l]) > 1e-8))
    nz_enet = int(np.sum(np.abs(enet_path[i_l]) > 1e-8))
    print(f"{a:>10.3f} {nz_ridge:>10} {nz_lasso:>10} {nz_enet:>13}")

print()
print("Ridge NEVER reaches zero - every coefficient stays alive at any alpha.")
print("Lasso's count falls in clean steps, like a variable selection procedure.")
print("ElasticNet sits between: sparse, but it keeps small groups of correlated")
print("features alive together, which Lasso tends to break arbitrarily.")


# ======================================================================================
# PART 4 - why Lasso struggles with correlated features, and ElasticNet fixes it
# ======================================================================================
print()
print("-" * 78)
print("PART 4 - the correlated-pair problem")
print("-" * 78)

# Build a dataset where features 0 and 1 are near-duplicates.
m = 300
base = rng.normal(0, 1, size=m)
X_corr = np.column_stack([
    base + rng.normal(0, 0.02, size=m),   # f0
    base + rng.normal(0, 0.02, size=m),   # f1 - essentially identical to f0
    rng.normal(0, 1, size=m),            # f2
    rng.normal(0, 1, size=m),            # f3
    rng.normal(0, 1, size=m),            # f4
])
y_corr = 5.0 * X_corr[:, 0] + 2.0 * X_corr[:, 2] + rng.normal(0, 1, size=m)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_corr, y_corr, test_size=0.3,
                                                         random_state=SEED)
print("f0 and f1 are the same variable to within 2% noise. The true coefficient")
print("on that latent variable is 5.0, split however the optimiser happens to choose.")
print()

print("Stability of the coefficients across 20 different random splits:")
print(f"{'model':<16} {'f0 mean':>9} {'f0 std':>9} {'f1 mean':>9} {'f1 std':>9} {'f0+f1':>9} {'test R2':>9}")
print("-" * 78)

for name, factory in [
    ("Ridge a=0.1", lambda: Ridge(alpha=0.1)),
    ("Lasso a=0.05", lambda: Lasso(alpha=0.05, max_iter=5000)),
    ("ElasticNet", lambda: ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000)),
]:
    coefs = []
    for seed in range(20):
        Xtr, Xte, ytr, yte = train_test_split(X_corr, y_corr, test_size=0.3,
                                              random_state=seed)
        mdl = factory().fit(Xtr, ytr)
        coefs.append(mdl.coef_)
    coefs = np.array(coefs)
    test_r2 = factory().fit(Xc_train, yc_train).score(Xc_test, yc_test)
    print(f"{name:<16} {coefs[:, 0].mean():>9.3f} {coefs[:, 0].std():>9.3f} "
          f"{coefs[:, 1].mean():>9.3f} {coefs[:, 1].std():>9.3f} "
          f"{(coefs[:, 0] + coefs[:, 1]).mean():>9.3f} {test_r2:>9.4f}")

print()
print("Lasso is UNSTABLE here: f0 gets 2.5 on one split and f1 gets it on the next,")
print("so both columns have a large standard deviation across refits. The SUM is")
print("stable at ~5.0 and the test score is fine - the model predicts correctly but")
print("cannot tell you which column to keep.")
print()
print("Ridge SPLITS the weight evenly and stably. ElasticNet also splits, because a")
print("large l1_ratio cannot separate near-identical columns. For correlated groups,")
print("grouping is better than either:")
print("  - ElasticNet with l1_ratio ~ 0.5-0.7")
print("  - group lasso / group ridge if scikit-learn's tooling fits your problem")
print("  - simply combine the columns into one feature (usually the best answer)")


# ======================================================================================
# PART 5 - automatic alpha selection
# ======================================================================================
print()
print("-" * 78)
print("PART 5 - let cross-validation pick alpha: the *CV classes")
print("-" * 78)

cv_detailed = RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED)
alpha_grid = np.logspace(-4, 4, 60)

ridge_cv_model = RidgeCV(alphas=alpha_grid, cv=cv_detailed).fit(X_train, y_train)
lasso_cv_model = LassoCV(cv=cv_detailed, n_alphas=100, max_iter=10000,
                         random_state=SEED).fit(X_train, y_train)
enet_cv_model = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                             n_alphas=60, cv=cv_detailed, max_iter=20000,
                             random_state=SEED).fit(X_train, y_train)

print(f"RidgeCV       best alpha = {ridge_cv_model.alpha_:.4f}   "
      f"test R2 = {ridge_cv_model.score(X_test, y_test):.4f}")
print(f"LassoCV       best alpha = {lasso_cv_model.alpha_:.5f}  "
      f"test R2 = {lasso_cv_model.score(X_test, y_test):.4f}")
print(f"ElasticNetCV  best alpha = {enet_cv_model.alpha_:.5f}  "
      f"l1_ratio = {enet_cv_model.l1_ratio_}   "
      f"test R2 = {enet_cv_model.score(X_test, y_test):.4f}")
print()
print("THE CRITICAL WARNING about every *CV class:")
print()
print("  They choose alpha using the SAME data you will report the score on.")
print("  That makes the reported R2 optimistically biased. For an honest number:")
print("    - nest the *CV search inside an outer CV loop, or")
print("    - use a fixed alpha from a separate validation set, or")
print("    - use GridSearchCV (same problem - solve it with nesting, script 10)")
print()
print("The *CV classes exist to save you from writing the grid loop, not to make")
print("your evaluation honest.")

selected = pd.DataFrame({
    "feature": feature_names,
    "ridge": Ridge(alpha=ridge_cv_model.alpha_).fit(X_train, y_train).coef_,
    "lasso": lasso_cv_model.coef_,
    "elastic_net": enet_cv_model.coef_,
    "truth": true_coef,
})
selected["lasso_kept"] = np.abs(selected["lasso"]) > 1e-8
print()
print("Lasso's automatic feature selection:")
print(selected[selected["feature"].isin(["f00", "f05", "f11"])].round(4).to_string(index=False))
kept = selected[selected["lasso_kept"]]["feature"].tolist()
print(f"\nFeatures Lasso kept: {len(kept)} -> {kept}")
print(f"The three real features are f00, f05, f11. Everything else was noise.")
print(f"Lasso recovered {sum(1 for f in kept if f in ['f00', 'f05', 'f11'])}/3 true features "
      f"and kept {len(kept) - 3} false positives.")
print()
print("With 120 samples and 20 features, a single tuning pass is already close to")
print("selection on noise. Cross-validation across repeats is what makes this")
print("trustworthy rather than lucky.")


# ======================================================================================
# PART 6 - the geometric picture
# ======================================================================================
print()
print("-" * 78)
print("PART 6 - why the shapes of the constraint regions matter")
print("-" * 78)
print("""
Minimise MSE subject to a constraint on w. The answer is the first point where
the MSE contour touches the constraint region.

  L2 / Ridge      ||w||_2 <= t      CIRCLE
  L1 / Lasso      ||w||_1 <= t      DIAMOND (corners lie on the axes)

        w2
         |   .-''''-.                 w2
         | .'  L1   '.               |     /\\
         |/           \\             |    /  \\   <- corners ON the axes
   ------+------.------             |   /____\\
         |       `.  L2 `            |  diamond
         |         `----'            |
                                   +----> w1

The L1 diamond's corners are exactly where one weight is zero and the rest are
not. A gradient path from the origin almost always hits a corner first, so the
solution is sparse.

The L2 circle touches NO axis. Every point on it has both weights non-zero, so
every solution is dense. Weights shrink smoothly toward zero and reach it only
in the limit alpha -> infinity.

  Practical consequences:
    - L1  -> automatic feature selection, interpretable sparse model
    - L2  -> better with many small effects, stable under collinearity
    - EN  -> the usual compromise; L2's stability with most of L1's sparsity
""")


# ======================================================================================
# Visualisation
# ======================================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# --- Plot 1: coefficient paths -------------------------------------------------------
ax = axes[0, 0]
for j in range(n_features):
    is_real = j in [0, 5, 11]
    ax.plot(alphas, ridge_path[:, j], color="#4C6EF5" if is_real else "#DEE2E6",
            linewidth=2.2 if is_real else 0.8, alpha=1.0 if is_real else 0.6)
for j in range(n_features):
    is_real = j in [0, 5, 11]
    ax.plot(alphas, lasso_path[:, j], color="#E03131" if is_real else "#FFA8A8",
            linewidth=2.2 if is_real else 0.8, alpha=1.0 if is_real else 0.6,
            linestyle="--" if is_real else "-")
ax.set_xscale("log")
ax.set_title("Coefficient paths: Ridge (solid) vs Lasso (dashed)", fontsize=11,
             fontweight="bold")
ax.set_xlabel("alpha (penalty strength)")
ax.set_ylabel("coefficient")
ax.plot([], [], color="#4C6EF5", linewidth=2, label="Ridge, real features")
ax.plot([], [], color="#E03131", linewidth=2, linestyle="--", label="Lasso, real features")
ax.plot([], [], color="#DEE2E6", linewidth=1, label="Ridge, noise features")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Plot 2: sparsity vs alpha -------------------------------------------------------
ax = axes[0, 1]
nz_ridge = (np.abs(ridge_path) > 1e-8).sum(axis=1)
nz_lasso = (np.abs(lasso_path) > 1e-8).sum(axis=1)
nz_enet = (np.abs(enet_path) > 1e-8).sum(axis=1)
ax.plot(alphas, nz_ridge, linewidth=2.2, color="#4C6EF5", label="Ridge (L2)")
ax.plot(alphas, nz_lasso, linewidth=2.2, color="#E03131", label="Lasso (L1)")
ax.plot(alphas, nz_enet, linewidth=2.2, color="#0CA678", label="ElasticNet")
ax.axhline(3, color="#212529", linestyle="--", linewidth=2, label="3 real features")
ax.set_xscale("log")
ax.set_title("Ridge never reaches zero. Lasso does.", fontsize=11, fontweight="bold")
ax.set_xlabel("alpha")
ax.set_ylabel("non-zero coefficients")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Plot 3: train vs test R2 as alpha grows -----------------------------------------
ax = axes[1, 0]
alpha_eval = np.logspace(-2, 2, 40)
for name, model in [("Ridge", Ridge), ("Lasso", Lasso), ("ElasticNet", ElasticNet)]:
    train_scores, test_scores = [], []
    for a in alpha_eval:
        pipe = Pipeline([("scale", StandardScaler()),
                         ("model", model(alpha=a, max_iter=5000))])
        pipe.fit(X_train, y_train)
        train_scores.append(pipe.score(X_train, y_train))
        test_scores.append(pipe.score(X_test, y_test))
    ax.plot(alpha_eval, train_scores, "--", linewidth=1.6, alpha=0.7, color="#868E96")
    ax.plot(alpha_eval, test_scores, linewidth=2.2, label=name)

ax.axvline(ridge_cv_model.alpha_, color="#4C6EF5", linestyle=":", linewidth=2)
ax.set_xscale("log")
ax.set_title("Dashed grey = training R2 (always falls too)", fontsize=11, fontweight="bold")
ax.set_xlabel("alpha")
ax.set_ylabel("R2")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Plot 4: the geometry ------------------------------------------------------------
ax = axes[1, 1]
t = 1.0
theta = np.linspace(0, 2 * np.pi, 400)
# L1 ball: |w1| + |w2| = t  ->  a diamond
l1_x = t * np.cos(theta) / (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))
l1_y = t * np.sin(theta) / (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))
# L2 ball: w1^2 + w2^2 = t^2  ->  a circle
l2_x, l2_y = t * np.cos(theta), t * np.sin(theta)

ax.plot(l2_x, l2_y, color="#4C6EF5", linewidth=2.5, label="L2 ball (Ridge) - circle")
ax.plot(l1_x, l1_y, color="#E03131", linewidth=2.5, label="L1 ball (Lasso) - diamond")
ax.scatter([0], [0], color="#212529", s=90, zorder=5, label="origin (OLS-ish start)")
ax.scatter([-t, 0], [0, -t], color="#E03131", s=120, marker="s", zorder=6,
           label="L1 corners: one weight is exactly 0")
ax.axhline(0, color="#ADB5BD", linewidth=1)
ax.axvline(0, color="#ADB5BD", linewidth=1)
ax.set_title("Why Lasso selects features and Ridge does not", fontsize=11, fontweight="bold")
ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.legend(fontsize=8, frameon=False, loc="upper right")
ax.grid(alpha=0.25)
ax.set_aspect("equal")

fig.suptitle("12 - Ridge, Lasso and Elastic Net", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
print("1. The penalty adds a bias/variance knob you control directly with alpha.")
print("   alpha = 0 is ordinary least squares; large alpha underfits.")
print("2. Ridge (L2) shrinks all coefficients smoothly and NEVER reaches zero.")
print("   Lasso (L1) drives some coefficients to EXACTLY zero - built-in feature")
print("   selection. The reason is the diamond vs circle constraint geometry.")
print("3. ElasticNet with l1_ratio ~ 0.5 is the pragmatic default when you want")
print("   sparsity but also stability with correlated features.")
print("4. Lasso is UNSTABLE with correlated features: it picks one arbitrarily and")
print("   the choice changes between refits. Prefer Ridge or ElasticNet there.")
print("5. Always STANDARDISE before regularising, or the penalty acts on units")
print("   rather than on importance.")
print("6. RidgeCV / LassoCV / ElasticNetCV choose alpha on the same data you report")
print("   on, so their scores are optimistic. Nest them if the number matters.")
print("7. alpha should never be tuned on a log scale below ~1e-4 for standardised")
print("   data - the model becomes indistinguishable from OLS.")
