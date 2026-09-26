"""
16 - Multicollinearity: When Features Disagree
==============================================

Goal: detect, quantify and fix the situation where several features carry the same
information and the model can no longer say which one matters.

The core insight
----------------
With perfectly collinear features, OLS has NO unique solution - the design matrix
is singular, and infinitely many coefficient vectors give identical predictions.
`pinv` returns one of them (the minimum-norm one). Predictions are fine.
Coefficients are arbitrary.

With NEARLY collinear features, OLS has a unique but wildly unstable solution.
Small changes in the data swing the coefficients enormously, because the fit is
trying to do something contradictory: satisfy two nearly-identical constraints at
once. The classic symptom is a large coefficient cancelling a large coefficient.

What it costs you
-----------------
  PREDICTIONS   fine (with a test set) - but degrades under distribution shift,
                because the extrapolation behaviour is governed by the unstable
                combination rather than by any real relationship
  COEFFICIENTS  unusable - you cannot say "distance matters more than area"
  CONFIDENCE    standard errors inflate, sometimes by 100x
  INFERENCE     significance tests become meaningless

The two standard measures
-------------------------
  Correlation   - pairwise, univariate, misses multicollinearity that arises only
                  from combinations of three or more features
  VIF           - variance of one coefficient explained by ALL the others.
                  VIF_j = 1 / (1 - R2_j) where R2_j comes from regressing x_j on
                  every other feature. That is why it catches combined effects.
                  Interpretation: VIF < 5 fine, 5-10 concerning, > 10 serious.

Run:  python 16_multicollinearity_detection.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
rng = np.random.default_rng(SEED)

print("=" * 78)
print("MULTICOLLINEARITY")
print("=" * 78)


# ======================================================================================
# PART 1 - build a dataset and hide a latent variable inside it
# ======================================================================================
print()
print("-" * 78)
print("PART 1 - the setup: three views of one underlying quantity")
print("-" * 78)

n = 500

# A single latent driver - think "true economic activity".
latent = rng.normal(0, 1, size=n)

# Three features are noisy measurements of that same latent variable.
gdp_per_capita = 15000 + 6000 * latent + rng.normal(0, 700, size=n)
median_income = 30000 + 7000 * latent + rng.normal(0, 900, size=n)
consumption = 20000 + 5000 * latent + rng.normal(0, 1200, size=n)

# Two genuinely independent features.
population = rng.uniform(0.5, 50, size=n)
unemployment = rng.uniform(2, 14, size=n)

# Consumption is the target, and it is driven almost entirely by the latent
# variable - so those three features are all 'right' and also redundant.
target = (0.55 * median_income + 0.9 * population - 180 * unemployment
          + rng.normal(0, 1500, size=n))

df = pd.DataFrame({
    "gdp_per_capita": gdp_per_capita,
    "median_income": median_income,
    "consumption": consumption,
    "population": population,
    "unemployment": unemployment,
})
features = ["gdp_per_capita", "median_income", "consumption", "population", "unemployment"]

print("Target: consumption.")
print("Three features (gdp, income, consumption) are all noisy measurements of the")
print("same latent variable, so they are all correlated with the target AND with")
print("each other. Two features (population, unemployment) are independent.")
print()
print("Pairwise correlations:")
print(df.corr(numeric_only=True).round(3).to_string())
print()
print("All three of gdp / income / consumption correlate ~0.85 with each other, yet")
print("pairwise correlation only shows you ONE relationship at a time. It cannot")
print("tell you that the correlation is explained by a shared latent factor.")
print()

X_train, X_test, y_train, y_test = train_test_split(
    df[features], target, test_size=0.25, random_state=SEED
)


# ======================================================================================
# PART 2 - VIF, computed by hand
# ======================================================================================
print()
print("-" * 78)
print("PART 2 - Variance Inflation Factor, from first principles")
print("-" * 78)
print("""
For each feature j:
  1. Regress x_j on ALL OTHER features.
  2. Record R2 of that auxiliary regression.
  3. VIF_j = 1 / (1 - R2_j)

Why this is the right definition: the standard error of coefficient j is inflated
by exactly sqrt(VIF_j) relative to what it would be if x_j were uncorrelated with
everything else. The shared variance among the correlated features gets divided
across all of them, and each one ends up with a large share of the estimation
error.

    SE_inflated = SE_uncorrelated * sqrt(VIF)
    CI_width    = proportional to SE
    t_statistic = coefficient / SE   -> shrinks as VIF grows
""")
print()

vif_values = {}
for feature in features:
    others = [f for f in features if f != feature]
    auxiliary = LinearRegression().fit(X_train[others], X_train[feature])
    r2_aux = auxiliary.score(X_train[others], X_train[feature])
    vif = 1 / (1 - r2_aux)
    vif_values[feature] = vif

print(f"{'feature':<20} {'aux R2':>9} {'VIF':>10} {'SE inflation':>14} {'verdict'}")
print("-" * 70)
for feature in features:
    r2_aux = 1 - 1 / vif_values[feature]
    vif = vif_values[feature]
    if vif > 10:
        verdict = "SEVERE - do not interpret this coefficient"
    elif vif > 5:
        verdict = "concerning"
    else:
        verdict = "acceptable"
    print(f"{feature:<20} {r2_aux:>9.4f} {vif:>10.2f} {np.sqrt(vif):>13.1f}x {verdict:>32}")

print()
print("`gdp_per_capita` can be predicted from the other four almost perfectly, so")
print("it has almost no unique variance left. Its standard error is inflated by")
print(f"{np.sqrt(vif_values['gdp_per_capita']):.0f}x, and any confidence interval or")
print("significance claim about it is worthless.")
print()

# Verify against statsmodels if it is available, because checking your own
# implementation against a reference implementation is a habit worth forming.
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm

    sm_vifs = [variance_inflation_factor(X_train.values, i) for i in range(len(features))]
    print("Verification against statsmodels.variance_inflation_factor:")
    for name, mine, theirs in zip(features, [vif_values[f] for f in features], sm_vifs):
        flag = "match" if np.isclose(mine, theirs, rtol=1e-6) else "MISMATCH"
        print(f"  {name:<20} mine {mine:>8.3f}   statsmodels {theirs:>8.3f}   {flag}")
    print()
except ImportError:
    print("(statsmodels not installed - the hand-computed VIFs above stand on their own)")
    print()


# ======================================================================================
# PART 3 - what multicollinearity does to the coefficients
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - the coefficient instability, measured")
print("-" * 78)

# Fit the same model on 25 different 75/25 splits and watch the coefficients move.
coef_runs = []
for split_seed in range(25):
    X_tr, X_te, y_tr, y_te = train_test_split(df[features], target, test_size=0.25,
                                              random_state=split_seed)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X_tr)
    m = LinearRegression().fit(scaled, y_tr)
    coef_runs.append(m.coef_)
coef_runs = np.array(coef_runs)

stability = pd.DataFrame({
    "feature": features,
    "coef_mean": coef_runs.mean(axis=0),
    "coef_std": coef_runs.std(axis=0),
    "min": coef_runs.min(axis=0),
    "max": coef_runs.max(axis=0),
    "sign_flips": [int(np.sum(np.sign(coef_runs[:, j]) != np.sign(coef_runs[:, j][0])))
                   for j in range(len(features))],
})
stability["range_over_std"] = (stability["max"] - stability["min"]) / stability["coef_std"]
print("Coefficients (standardised data) across 25 random refits:")
print(stability.round(3).to_string(index=False))
print()
print("The three collinear features have huge standard deviations - often larger")
print("than the coefficients themselves. That instability is collinearity, and it")
print("means the model cannot say which of the three is doing the work.")
print()
print("Compare with population and unemployment: small coefficients, small")
print("standard deviations, stable signs. Their information is uniquely theirs.")
print()
print("IMPORTANT: test scores are essentially IDENTICAL across all these models.")
print("Collinearity damages EXPLANATION, not PREDICTION (on data like the training")
print("distribution). That asymmetry is why the problem hides so well.")
print()


# ======================================================================================
# PART 4 - the fixes, measured
# ======================================================================================
print()
print("-" * 78)
print("PART 4 - four fixes, compared")
print("-" * 78)

cv_strategy = 5

variants = {}
variants["all 5 features"] = features
variants["drop gdp_per_capita"] = [f for f in features if f != "gdp_per_capita"]
variants["drop all 3 collinear"] = ["population", "unemployment"]
variants["keep only median_income"] = ["median_income", "population", "unemployment"]

print(f"{'variant':<28} {'#feat':>7} {'max VIF':>9} {'CV R2':>9} {'test R2':>9} {'coef range':>12}")
print("-" * 82)

for name, cols in variants.items():
    # Work in plain NumPy here. A StandardScaler fitted on ALL columns refuses to
    # transform a SUBSET of them, because it checks the feature names it saw at
    # fit time. Vising is a per-feature calculation, so NumPy avoids the whole
    # issue and is clearer.
    X_tr = X_train[cols].to_numpy()
    X_te = X_test[cols].to_numpy()
    scaler = StandardScaler().fit(X_tr)
    m = LinearRegression().fit(scaler.transform(X_tr), y_train)

    # VIF of column j = 1 / (1 - R2 of column j regressed on the other columns).
    vifs = []
    for j in range(len(cols)):
        other_indices = [k for k in range(len(cols)) if k != j]
        aux = LinearRegression().fit(X_tr[:, other_indices], X_tr[:, j].ravel())
        vifs.append(1 / (1 - aux.score(X_tr[:, other_indices], X_tr[:, j].ravel())))

    pipe = Pipeline([("scale", StandardScaler()), ("m", LinearRegression())])
    cv_r2 = cross_val_score(pipe, X_train[cols], y_train, cv=cv_strategy, scoring="r2").mean()
    test_r2 = pipe.fit(X_train[cols], y_train).score(X_test[cols], y_test)

    # Stability: how far the coefficients move across 15 refits.
    runs = []
    for seed in range(15):
        a, b, c, d = train_test_split(df[cols], target, test_size=0.25, random_state=seed)
        s = StandardScaler().fit(a)
        runs.append(LinearRegression().fit(s.transform(a), c).coef_)
    coef_range = np.max(np.std(np.array(runs), axis=0))

    print(f"{name:<28} {len(cols):>7} {max(vifs):>9.2f} {cv_r2:>9.4f} {test_r2:>9.4f} "
          f"{coef_range:>12.2f}")

print("-" * 82)
print()
print("Key observations:")
print()
print("1. Dropping the redundant features barely changes CV R2 or test R2. The model")
print("   was not using them; they were substitutes for one another.")
print("2. It collapses the max VIF and the coefficient range. THAT is the payoff:")
print("   an interpretable, stable model at essentially no predictive cost.")
print("3. 'Drop all 3 collinear' throws away the strongest signal in the data. It is")
print("   the WORST choice here - so drop collinear features thoughtfully, keeping")
print("   the one most central to your question, not blindly deleting all of them.")
print()


# ======================================================================================
# PART 5 - Ridge as the automatic fix
# ======================================================================================
print()
print("-" * 78)
print("PART 5 - Ridge: shrinking correlated features instead of choosing between them")
print("-" * 78)
print("Ridge does not choose one of the correlated features. It spreads the weight")
print("across all of them, which turns an unstable estimate into a stable - and")
print("less interpretable - one. That is often the right trade.")
print()
print(f"{'model':<26} {'alpha':>8} {'CV R2':>9} {'coef_std_across_refits':>25}")
print("-" * 72)

for name, model in [("OLS", LinearRegression()), ("Ridge", Ridge)]:
    for alpha in ([0] if name == "OLS" else [0.5, 5.0, 50.0, 200.0]):
        pipe = Pipeline([("scale", StandardScaler()), ("m", Ridge(alpha=alpha))])
        cv_r2 = cross_val_score(pipe, X_train, y_train, cv=cv_strategy, scoring="r2").mean()

        runs = []
        for seed in range(15):
            # train_test_split returns [X_train, X_test, y_train, y_test] in that
            # order. Skipping the test features in the middle with `_` keeps the
            # train/target pairing correct - easy to get wrong.
            a, _, c, _d = train_test_split(df[features], target, test_size=0.25,
                                            random_state=seed)
            s = StandardScaler().fit(a)
            runs.append(Ridge(alpha=alpha).fit(s.transform(a), c).coef_)
        instability = np.mean(np.std(np.array(runs), axis=0))

        print(f"{name:<26} {alpha:>8} {cv_r2:>9.4f} {instability:>25.3f}")

print()
print("As alpha grows, the coefficients stop swinging. CV R2 barely moves, because")
print("Ridge is right that the information content is unchanged - it just stops")
print("pretending it can attribute it to one column rather than another.")
print()

# Show the coefficient SPLIT for the correlated trio.
print("How Ridge divides the weight across the three collinear features:")
print(f"{'alpha':>8} {'gdp':>10} {'income':>10} {'consumption':>14} {'sum':>9}")
print("-" * 56)
scaler = StandardScaler().fit(X_train)
X_scaled = scaler.transform(X_train)
for alpha in [0.0, 1.0, 10.0, 100.0, 500.0]:
    m = Ridge(alpha=alpha).fit(X_scaled, y_train)
    trio = m.coef_[:3]
    print(f"{alpha:>8.1f} {trio[0]:>10.2f} {trio[1]:>10.2f} {trio[2]:>14.2f} "
          f"{trio.sum():>9.2f}")
print()
print("The SUM of the three is much more stable than any individual coefficient.")
print("That sum is the honest, reportable quantity: 'combined macroeconomic")
print("strength carries this much weight'. Ridge is effectively telling you that the")
print("data supports the latent factor, and cannot support the individual proxies.")


# ======================================================================================
# PART 6 - VIF as an automatic elimination procedure
# ======================================================================================
print()
print("-" * 78)
print("PART 6 - iterative VIF elimination")
print("-" * 78)


def calculate_vif(dataframe):
    """VIF for every column of a DataFrame, computed by auxiliary regression."""
    vifs = {}
    for column in dataframe.columns:
        others = [c for c in dataframe.columns if c != column]
        if not others:
            vifs[column] = 1.0
            continue
        x = dataframe[others].values
        z = dataframe[column].values
        aux = LinearRegression().fit(x, z)
        r2 = aux.score(x, z)
        vifs[column] = 1 / (1 - r2) if r2 < 1 else np.inf
    return pd.Series(vifs, name="VIF")


remaining = list(features)
elimination_log = []
threshold = 5.0

print(f"Eliminating features with VIF > {threshold}, one at a time, re-checking after each removal:")
print(f"{'step':>5} {'removed':<20} {'remaining':<50} {'new max VIF':>12}")
print("-" * 90)

step = 0
while len(remaining) > 1:
    vifs = calculate_vif(X_train[remaining])
    worst = vifs.idxmax()
    if vifs[worst] <= threshold:
        print(f"{step:>5} {'(none - all VIFs acceptable)':<20} {str(remaining):<50} "
              f"{vifs.max():>12.2f}")
        break
    remaining.remove(worst)
    step += 1
    new_vifs = calculate_vif(X_train[remaining])
    elimination_log.append({"step": step, "removed": worst, "max_vif_after": new_vifs.max()})
    print(f"{step:>5} {worst:<20} {str(remaining):<50} {new_vifs.max():>12.2f}")

print()
print(f"Survivors: {remaining}")
print()
print("The order of elimination matters, and it is arbitrary. Ties at similar VIF")
print("mean the procedure is picking between two equally redundant features. That")
print("is a good signal to stop and make a domain decision rather than let an")
print("algorithm decide which variable your report will feature.")
print()
print("Also: VIF is only meaningful for INDEPENDENT observations. With time series")
print("or clustered data, apparent collinearity can be a lag structure instead, and")
print("deleting those features removes genuine signal.")


# ======================================================================================
# PART 7 - the recommendation
# ======================================================================================
print()
print("-" * 78)
print("PART 7 - decision guide")
print("-" * 78)
print("""
Ask yourself two questions, in this order.

Q1: Do I need to EXPLAIN the model, or only PREDICT with it?

    Only predict          -> collinearity is mostly a non-issue. Keep everything,
                             add Ridge, use the extra features as a hedge against
                             any one of them drifting. Cross-validation decides.

    Must explain          -> fix it. Drop to an interpretable subset, report the
                             VIFs, and state which features were excluded and why.

Q2: If I must explain, do I need THIS feature, or the underlying thing it measures?

    gdp_per_capita, median_income, consumption all proxy "economic strength".
    Pick ONE representative, name it clearly, and note that the others were
    excluded as redundant. That is a better report than three unstable
    coefficients that all say the same thing.

Fixes ranked by how often they are the right answer:
    1. Drop the redundant column and say why           <- clearest, most honest
    2. Combine them deliberately (mean, ratio, index)  <- best when they are one concept
    3. Ridge, if you need all of them for prediction   <- stable, less interpretable
    4. PCA on the correlated block                     <- when there are many
    5. Ignore it                                      <- acceptable if you only predict

What NOT to do:
    - Delete features purely because their p-value is high. With collinearity the
      p-values are unreliable in BOTH directions, and you may discard the only
      feature that carries the signal.
    - Delete features to make the residual plots look better. The plot is not
      broken; the specification is.
    - Trust a coefficient sign that changes between refits. If it does, the sign
      is not a finding.
""")


# ======================================================================================
# Visualisation
# ======================================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# --- Plot 1: correlation heatmap -------------------------------------------------------
ax = axes[0, 0]
corr = X_train.corr(numeric_only=True)
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(features)))
ax.set_yticks(range(len(features)))
ax.set_xticklabels(features, rotation=40, ha="right", fontsize=8)
ax.set_yticklabels(features, fontsize=8)
for i in range(len(features)):
    for j in range(len(features)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
ax.set_title("Pairwise correlation: shows pairs, not the pattern", fontsize=11,
             fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.046)

# --- Plot 2: VIF bars ------------------------------------------------------------------
ax = axes[0, 1]
vif_series = pd.Series(vif_values).sort_values(ascending=False)
bar_colors = ["#E03131" if v > 10 else "#F59F00" if v > 5 else "#0CA678"
              for v in vif_series]
ax.bar(range(len(vif_series)), vif_series.values, color=bar_colors)
ax.axhline(5, color="#F59F00", linestyle="--", linewidth=2, label="VIF = 5 (concerning)")
ax.axhline(10, color="#E03131", linestyle="--", linewidth=2, label="VIF = 10 (serious)")
ax.set_xticks(range(len(vif_series)))
ax.set_xticklabels(vif_series.index, rotation=40, ha="right", fontsize=8)
ax.set_title("Variance Inflation Factor by feature", fontsize=11, fontweight="bold")
ax.set_ylabel("VIF")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25, axis="y")

# --- Plot 3: coefficient instability ---------------------------------------------------
ax = axes[1, 0]
positions = np.arange(len(features))
for j, feature in enumerate(features):
    ax.scatter(np.full(25, positions[j]) + rng.normal(0, 0.07, 25), coef_runs[:, j],
               s=22, alpha=0.75, color="#E03131" if vif_values[feature] > 5 else "#0CA678",
               edgecolors="none")
ax.axhline(0, color="#212529", linewidth=1.5)
ax.set_xticks(positions)
ax.set_xticklabels(features, rotation=40, ha="right", fontsize=8)
ax.set_title("Each dot is one refit (red = high VIF)", fontsize=11, fontweight="bold")
ax.set_ylabel("standardised coefficient")
ax.grid(alpha=0.25, axis="y")

# --- Plot 4: Ridge coefficient split ---------------------------------------------------
ax = axes[1, 1]
alphas_ridge = np.logspace(-1, 3, 40)
paths = np.array([Ridge(alpha=a).fit(X_scaled, y_train).coef_[:3] for a in alphas_ridge])
for j, feature in enumerate(features[:3]):
    ax.plot(alphas_ridge, paths[:, j], linewidth=2.2, label=feature)
ax.plot(alphas_ridge, paths.sum(axis=1), color="#212529", linewidth=2.8,
        linestyle="--", label="sum of the three")
ax.set_xscale("log")
ax.set_title("Ridge stabilises the trio, and their sum was always stable",
             fontsize=11, fontweight="bold")
ax.set_xlabel("alpha")
ax.set_ylabel("coefficient")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

fig.suptitle("16 - Multicollinearity", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
print("1. Multicollinearity breaks EXPLANATION, not prediction. Test scores stay")
print("   the same while coefficients swing wildly between refits. That asymmetry")
print("   is why it goes unnoticed for so long.")
print("2. VIF = 1/(1 - R2 of x_j regressed on all other features). It catches")
print("   collinearity arising from combinations, which pairwise correlation cannot.")
print("3. Thresholds: VIF < 5 fine, 5-10 concerning, > 10 do not interpret the")
print("   coefficient. A feature's standard error is inflated by exactly sqrt(VIF).")
print("4. Fix in order of preference: drop the redundant column and document it;")
print("   combine the columns into one concept; apply Ridge; apply PCA. Ignoring it")
print("   is defensible when you only need predictions.")
print("5. Never drop a feature because its p-value is high. Under collinearity the")
print("   p-values are unreliable in both directions.")
print("6. A coefficient whose sign changes between refits is not a finding. Check")
print("   this by refitting on a handful of different splits before you report it.")
