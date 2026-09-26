"""
14 - Feature Importance and Feature Selection
=============================================

Goal: work out which features actually earn their place, and remove the rest
without damaging the model.

Two different questions, two different tools
---------------------------------------------
1. "How much does this feature contribute to the PREDICTIONS?"
   -> permutation importance, and coefficients on standardised data

2. "Which features should I KEEP in the model?"
   -> RFE, SelectKBest, L1-based selection, VIF (script 16)

These are different questions. A feature can be individually uninformative and
still matter, because it is redundant with another feature. Conversely, a feature
can have a large coefficient and be worthless to remove, because something else
substitutes for it.

The methods compared here
-------------------------
| Method                        | Type        | Needs a model? | Handles redundancy? |
|-------------------------------|-------------|----------------|---------------------|
| Standardised coefficient       | intrinsic   | no             | no                  |
| |t| statistic                  | intrinsic   | no             | no                  |
| RFE                           | wrapper     | yes            | partly              |
| SelectKBest (f_regression)    | filter      | no             | no                  |
| SelectKBest (mutual_info)     | filter      | no             | no                  |
| Lasso                         | embedded    | yes            | yes, by choosing   |
| Permutation importance        | model-agnostic | no (needs a fitted model) | yes |

Run:  python 14_feature_importance_and_selection.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    f_regression,
    mutual_info_regression,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
rng = np.random.default_rng(SEED)

print("=" * 78)
print("FEATURE IMPORTANCE AND FEATURE SELECTION")
print("=" * 78)


# ======================================================================================
# PART 1 - a dataset where we know the answer
# ======================================================================================
print()
print("-" * 78)
print("PART 1 - 25 features, 3 real, 4 redundant, 18 pure noise")
print("-" * 78)

n_samples = 300
n_features = 25

signal = rng.normal(0, 1, size=(n_samples, 3))
# Features 0-2 are the real signal.
# Features 3-6 are noisy copies of feature 0 - redundant, not new information.
# Features 7-24 are pure noise.
redundant = np.column_stack([
    signal[:, 0] * 0.9 + rng.normal(0, 0.4, size=n_samples) for _ in range(4)
])
noise = rng.normal(0, 1, size=(n_samples, n_features - 7))
X = np.hstack([signal, redundant, noise])
y = 6.0 * signal[:, 0] - 3.0 * signal[:, 1] + 2.0 * signal[:, 2] + rng.normal(0, 1.0, size=n_samples)

feature_names = (
    [f"signal_{i}" for i in range(3)]
    + [f"redundant_{i}" for i in range(4)]
    + [f"noise_{i:02d}" for i in range(n_features - 7)]
)
X_df = pd.DataFrame(X, columns=feature_names)
print(f"{n_samples} samples x {n_features} features")
print("  signal_0..2  : real signal, true coefs 6.0, -3.0, 2.0")
print("  redundant_0..3: 90% copies of signal_0, plus independent noise")
print("  noise_00..17 : nothing at all")
print()
print("The redundant block is the interesting part. Every one of those columns is")
print("correlated with the target, so a univariate filter will happily keep them all.")
print("They add nothing, and they destabilise the coefficients.")

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.3, random_state=SEED)


# ======================================================================================
# PART 2 - intrinsic importance: coefficients and t-statistics
# ======================================================================================
print()
print("-" * 78)
print("PART 2 - intrinsic importance (no refitting needed)")
print("-" * 78)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
ols = LinearRegression().fit(X_train_s, y_train)

residuals = y_train - ols.predict(X_train_s)
n_obs, p = X_train_s.shape
dof = n_obs - p - 1
sigma_sq = np.sum(residuals**2) / dof
cov = sigma_sq * np.linalg.pinv(X_train_s.T @ X_train_s)
std_errors = np.sqrt(np.diag(cov))
t_stats = ols.coef_ / std_errors

intrinsic = pd.DataFrame({
    "feature": feature_names,
    "coef_standardised": ols.coef_,
    "std_error": std_errors,
    "t_stat": t_stats,
    "true_group": ["signal"] * 3 + ["redundant"] * 4 + ["noise"] * 18,
}).sort_values("t_stat", key=np.abs, ascending=False)

print(intrinsic.head(10).round(3).to_string(index=False))
print("  ...")
print()
print("The four redundant columns all show large |t|. That is CORRECT - they are")
print("genuinely correlated with y. It just does not mean they are all needed.")
print("Multivariate methods must break the tie; univariate ones cannot see the problem.")
print()
print(f"Mean |t| by group:")
print(intrinsic.groupby("true_group")["t_stat"].apply(lambda s: s.abs().mean()).round(3).to_string())
print()
print("Noise features have high |t| too. With 18 pure noise features and n = 210,")
print("two of them showing |t| > 2 is not surprising - that is what chance does.")


# ======================================================================================
# PART 3 - filter methods
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - filter methods: rank by a single-feature statistic")
print("-" * 78)

f_scores, f_pvalues = f_regression(X_train, y_train)
mi_scores = mutual_info_regression(X_train, y_train, random_state=SEED)

filter_table = pd.DataFrame({
    "feature": feature_names,
    "group": intrinsic["true_group"].values,
    "F_score": f_scores,
    "F_pvalue": f_pvalues,
    "mutual_info": mi_scores,
})
print("Top 10 by F-score:")
print(filter_table.sort_values("F_score", ascending=False).head(10).round(3).to_string(index=False))
print()
print("Top 10 by mutual information:")
print(filter_table.sort_values("mutual_info", ascending=False).head(10).round(3).to_string(index=False))
print()

for k in [3, 5, 8, 12]:
    top_f = list(filter_table.sort_values("F_score", ascending=False)["feature"].head(k))
    n_real = sum(1 for f in top_f if f.startswith("signal"))
    n_red = sum(1 for f in top_f if f.startswith("redundant"))
    n_noise = sum(1 for f in top_f if f.startswith("noise"))
    print(f"  SelectKBest(k={k:>2}) F-score  : {n_real} real, {n_red} redundant, {n_noise} noise")
print()
print("The filter reliably finds the real features. It cannot tell a redundant")
print("column from an informative one, so the top k is padded with duplicates.")


# ======================================================================================
# PART 4 - embedded selection: Lasso
# ======================================================================================
print()
print("-" * 78)
print("PART 4 - embedded selection: let L1 pick the subset")
print("-" * 78)

print(f"{'alpha':>9} {'#kept':>7} {'features kept':>52} {'test R2':>9}")
print("-" * 82)

best_lasso = None
for alpha in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.5]:
    lasso = Lasso(alpha=alpha, max_iter=20000).fit(X_train_s, y_train)
    kept = [feature_names[i] for i in range(n_features) if abs(lasso.coef_[i]) > 1e-8]
    test_r2 = lasso.score(scaler.transform(X_test), y_test)
    shown = ", ".join(kept[:4]) + (f" ... +{len(kept) - 4}" if len(kept) > 4 else "")
    print(f"{alpha:>9} {len(kept):>7} {shown:>52} {test_r2:>9.4f}")
    if best_lasso is None or test_r2 > best_lasso[1]:
        best_lasso = (lasso, test_r2, alpha)

lasso_best, r2_best, alpha_best = best_lasso
kept_best = [feature_names[i] for i in range(n_features) if abs(lasso_best.coef_[i]) > 1e-8]
print("-" * 82)
print(f"Best Lasso alpha {alpha_best}: kept {len(kept_best)} features -> {kept_best}")
print()
print("Compare the coefficient values. The redundant block gets a FRACTION of the")
print("weight, split across four columns, while signal_0 keeps a large share. Lasso")
print("resolves redundancy by dividing the weight, which is a sensible default")
print("behaviour - and one that is stable enough to act on.")
print()
print("This is the difference from a filter: the redundancy was judged jointly, not")
print("one column at a time.")


# ======================================================================================
# PART 5 - wrapper methods: RFE
# ======================================================================================
print()
print("-" * 78)
print("PART 5 - wrapper methods: RFE refits the model for every candidate subset")
print("-" * 78)
print("RFE is expensive and powerful: it repeatedly trains the model, strips the")
print("least important feature, and keeps the subset size that scores best.")
print()

cv_strategy = KFold(5, shuffle=True, random_state=SEED)
print(f"{'k kept':>8} {'CV R2 (RFE)':>13} {'n_features':>12} {'selected (first 6)':>44}")
print("-" * 82)

rfe_results = {}
for k in [3, 5, 8, 12, 18]:
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=k, step=1)
    rfe.fit(X_train, y_train)
    scores = cross_val_score(
        Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
        X_train.iloc[:, rfe.support_], y_train, cv=cv_strategy, scoring="r2"
    )
    selected = [f for f, keep in zip(feature_names, rfe.support_) if keep]
    rfe_results[k] = (scores.mean(), selected)
    print(f"{k:>8} {scores.mean():>13.4f} {k:>12} {', '.join(selected[:6]):>44}")

best_k = max(rfe_results, key=lambda k: rfe_results[k][0])
print("-" * 82)
print(f"RFE's best subset size: {best_k} -> {rfe_results[best_k][1]}")
n_real = sum(1 for f in rfe_results[best_k][1] if f.startswith("signal"))
print(f"Real features recovered: {n_real}/3")


# ======================================================================================
# PART 6 - permutation importance: what actually matters for the PREDICTIONS
# ======================================================================================
print()
print("-" * 78)
print("PART 6 - permutation importance (model-agnostic, measures real contribution)")
print("-" * 78)
print("Shuffle ONE column at random and see how much the score drops. The drop is")
print("that feature's importance. Repeat over many shuffles for a reliable estimate.")
print()

full_pipe = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
full_pipe.fit(X_train, y_train)
base_r2 = full_pipe.score(X_test, y_test)
print(f"Baseline test R2 with all {n_features} features: {base_r2:.4f}")
print()
perm = permutation_importance(
    full_pipe, X_test, y_test, n_repeats=30, random_state=SEED, scoring="r2"
)
perm_table = pd.DataFrame({
    "feature": feature_names,
    "group": intrinsic["true_group"].values,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)

print("Top 12 by permutation importance:")
print(perm_table.head(12).round(4).to_string(index=False))
print()
print(f"Mean importance by group:")
print(perm_table.groupby("group")["importance_mean"].mean().round(4).to_string())
print()
print("Permutation importance on the TEST set captures redundancy correctly: the")
print("redundant columns score low, because shuffling any one of them barely moves")
print("the prediction - the other three cover for it. No univariate method can")
print("see this, and it is the strongest argument for permutation importance.")
print()
print(f"Total importance: {perm.importances_mean.sum():.4f}  "
      f"(compare to the baseline R2 of {base_r2:.4f})")
print("Importance values for the noise features hover around 0 or slightly negative -")
print("negative means the model did BETTER with that column shuffled, which is pure")
print("noise. Treating negative importance as 'useless' is the correct reading.")


# ======================================================================================
# PART 7 - putting it together
# ======================================================================================
print()
print("-" * 78)
print("PART 7 - does selection actually improve the model?")
print("-" * 78)
print("Feature selection is often sold as an accuracy win. It is usually a")
print("VARIANCE and INTERPRETABILITY win. Measure it honestly:")
print()

full_cv = cross_val_score(full_pipe, X_train, y_train, cv=cv_strategy, scoring="r2").mean()

selected_variants = {}
for k in [3, 5, 8, 12]:
    rfe_k = RFE(estimator=LinearRegression(), n_features_to_select=k, step=1)
    rfe_k.fit(X_train, y_train)
    cols = [f for f, keep in zip(feature_names, rfe_k.support_) if keep]
    pipe = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    cv_score = cross_val_score(pipe, X_train[cols], y_train, cv=cv_strategy,
                               scoring="r2").mean()
    pipe.fit(X_train[cols], y_train)
    test_r2 = pipe.score(X_test[cols], y_test)
    test_mae = mean_absolute_error(y_test, pipe.predict(X_test[cols]))
    selected_variants[k] = (cols, cv_score, test_r2, test_mae)

print(f"{'variant':<30} {'CV R2':>8} {'test R2':>9} {'test MAE':>10}")
print("-" * 60)
print(f"{'all 25 features':<30} {full_cv:>8.4f} {base_r2:>9.4f} "
      f"{mean_absolute_error(y_test, full_pipe.predict(X_test)):>10.4f}")
for k, (cols, cv_s, t_s, t_mae) in selected_variants.items():
    print(f"{f'RFE top {k}':<30} {cv_s:>8.4f} {t_s:>9.4f} {t_mae:>10.4f}")

print()
print("Typical outcome: CV score is flat or slightly better, test score is within")
print("noise, and the model is dramatically simpler. That is still worth having -")
print("fewer inputs means less data collection, faster inference, and an explanation")
print("a human can read.")
print()
print("NOTE: RFE here is fitted on X_train and the resulting columns are then scored")
print("by CV on the same X_train. That is mildly optimistic. In production, put the")
print("selector INSIDE a pipeline so it is refitted within every fold:")
print()
print("    pipe = Pipeline([('scale', StandardScaler()),")
print("                     ('select', SelectFromModel(Lasso(alpha=0.3))),")
print("                     ('model', LinearRegression())])")
print("    cross_val_score(pipe, X, y, cv=5)      # no leakage: selection is refit")


# ======================================================================================
# Visualisation
# ======================================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# --- Plot 1: all three importance measures side by side -------------------------------
ax = axes[0, 0]
comparison = pd.DataFrame({
    "feature": feature_names,
    "t_abs": np.abs(intrinsic.set_index("feature").loc[feature_names, "t_stat"].to_numpy()),
    "f_score": filter_table.set_index("feature").loc[feature_names, "F_score"].to_numpy(),
    "perm": perm_table.set_index("feature").loc[feature_names, "importance_mean"].to_numpy(),
}).sort_values("perm", ascending=False)
positions = np.arange(len(comparison))
ax.barh(positions - 0.25, comparison["t_abs"], 0.25, label="|t| statistic", color="#868E96")
ax.barh(positions, comparison["f_score"], 0.25, label="F-score", color="#4C6EF5")
ax.barh(positions + 0.25, comparison["perm"], 0.25, label="permutation", color="#E03131")
ax.set_yticks(positions)
ax.set_yticklabels(comparison["feature"], fontsize=7)
ax.invert_yaxis()
ax.set_title("Three notions of importance disagree", fontsize=11, fontweight="bold")
ax.set_xlabel("importance (normalised per measure is NOT possible - compare ranks)")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25, axis="x")

# --- Plot 2: permutation importance by group ------------------------------------------
ax = axes[0, 1]
grouped = perm_table.groupby("group")["importance_mean"].describe()[["mean", "50%", "max"]]
grouped.plot(kind="bar", ax=ax, width=0.7, color=["#4C6EF5", "#0CA678", "#E03131"])
ax.axhline(0, color="#212529", linewidth=1)
ax.set_title("Permutation importance separates real from redundant", fontsize=11,
             fontweight="bold")
ax.set_xlabel("feature group")
ax.set_ylabel("mean importance")
ax.tick_params(axis="x", labelrotation=0)
ax.legend(fontsize=8, frameon=False, title="statistic")
ax.grid(alpha=0.25, axis="y")

# --- Plot 3: how the top-k list changes with k -----------------------------------------
ax = axes[1, 0]
f_ranked = filter_table.sort_values("F_score", ascending=False)["feature"].tolist()
k_values = list(range(1, 26))
f_counts = [sum(1 for f in f_ranked[:k] if f.startswith("signal")) for k in k_values]
r_counts = [sum(1 for f in f_ranked[:k] if f.startswith("redundant")) for k in k_values]
n_counts = [sum(1 for f in f_ranked[:k] if f.startswith("noise")) for k in k_values]
ax.stackplot(k_values, f_counts, r_counts, n_counts,
             labels=["signal (real)", "redundant (duplicates)", "noise (useless)"],
             colors=["#0CA678", "#F59F00", "#DEE2E6"])
ax.set_title("What a F-score top-k list contains", fontsize=11, fontweight="bold")
ax.set_xlabel("k (features selected)")
ax.set_ylabel("count")
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.grid(alpha=0.25)

# --- Plot 4: CV score by subset size ---------------------------------------------------
ax = axes[1, 1]
ks = sorted(rfe_results)
cv_vals = [rfe_results[k][0] for k in ks]
ax.plot(ks, cv_vals, "o-", color="#4C6EF5", linewidth=2.2, label="CV R2 by subset size")
ax.axhline(full_cv, color="#E03131", linestyle="--", linewidth=2,
           label=f"all features ({full_cv:.4f})")
ax.axvline(best_k, color="#0CA678", linestyle="--", linewidth=2,
           label=f"RFE best k = {best_k}")
ax.set_title("Selection: small gain in score, large gain in simplicity", fontsize=11,
             fontweight="bold")
ax.set_xlabel("number of features kept")
ax.set_ylabel("cross-validated R2")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

fig.suptitle("14 - Feature Importance and Selection", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
print("1. 'Which features matter' has two answers. Coefficients and t-stats say what")
print("   the model did with the features. Permutation importance says what the")
print("   features are worth for the predictions. Answer both questions.")
print("2. Univariate filters (F-score, mutual info) cannot see redundancy. A column")
print("   that duplicates another will rank just as highly and add nothing.")
print("3. Embedded L1 selection and RFE judge features JOINTLY, so they can drop")
print("   duplicates. That is why they beat filters on redundant data.")
print("4. Permutation importance is the only method here that correctly scores a")
print("   redundant column as unimportant, because shuffling it changes nothing.")
print("5. Negative permutation importance means 'the model was slightly better")
print("   without it' - i.e. pure noise. Treat it as zero.")
print("6. Feature selection usually buys variance reduction and interpretability,")
print("   not raw accuracy. Measure it, and expect a small CV gain.")
print("7. Put selectors INSIDE the pipeline. Fitting them before cross_val_score")
print("   leaks the test fold into the selection and inflates the score.")
print("8. Compare against a RandomForest's importances when a single tree family")
print("   keeps giving a different answer - two very different estimators agreeing")
print("   is meaningful evidence, one of them is not.")
