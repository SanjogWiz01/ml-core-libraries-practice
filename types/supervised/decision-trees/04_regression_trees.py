"""
04 - Regression Trees
=====================

Goal: understand `DecisionTreeRegressor` - how variance replaces impurity, what
a leaf actually predicts, and why regression trees make piecewise-constant
predictions that linear models cannot imitate.

The one-line difference from classification
-------------------------------------------
Classification splits nodes to reduce IMPURITY (Gini or entropy). Regression
splits nodes to reduce VARIANCE:

    variance(node) = (1/n) * SUM_i (y_i - mean(y))^2

This is the mean squared error of predicting the node's mean, so a regression
tree is doing ordinary least squares recursively, one split at a time. The
greedy structure is identical; only the objective changes.

An identity worth knowing
-------------------------
For a node of size n, the sum of squared deviations decomposes as

    SUM (y_i - mean)^2 = SUM (y_i - c)^2 - n * (mean - c)^2     for any constant c

So for a split into L and R, the reduction in squared error is exactly

    SSE(parent) - SSE(left) - SSE(right)

Maximising that is minimising the variance reduction, which is why the criterion
is called 'squared_error' and why MSE and RMSE appear in its documentation - they
are the same objective up to a monotone rescaling, exactly as log_loss and
entropy were the same in script 03.

What a leaf predicts
--------------------
The MEAN of the training targets that reached it. So a regression tree's
prediction function is piecewise constant: a staircase. It can never extrapolate
beyond the target range it saw, and it buys curvature with segments rather than
with coefficients - which is why depth, not feature count, is the knob that
controls how much detail a tree can fit.

Run:  python 04_regression_trees.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text

SEED = 42

print("=" * 78)
print("04 - REGRESSION TREES")
print("=" * 78)


# ======================================================================================
# Data: diabetes, small enough to reason about, real enough to be interesting
# ======================================================================================
print()
print("-" * 78)
print("DATA")
print("-" * 78)

diabetes_X, diabetes_y = load_diabetes(return_X_y=True)
feature_names = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
feature_names = [f"{n}_std" for n in feature_names]

X_train, X_test, y_train, y_test = train_test_split(
    diabetes_X, diabetes_y, test_size=0.25, random_state=SEED
)
print(f"{diabetes_X.shape[0]} rows x {diabetes_X.shape[1]} features")
print(f"target: disease progression, mean {diabetes_y.mean():.1f}, "
      f"sd {diabetes_y.std():.1f}")
print(f"train {len(X_train)} / test {len(X_test)}")
print()
print("scikit-learn already standardises these features, which means they are all on")
print("a comparable scale. That is convenient, but script 05 shows trees would not")
print("have needed it - a useful property worth knowing when your data is not")
print("pre-scaled.")
print()

# Constant prediction is the baseline any regression model must beat.
mean_baseline_r2 = 1.0 - np.sum((y_test - y_train.mean()) ** 2) / np.sum(
    (y_test - y_test.mean()) ** 2)
print(f"Baseline - always predict the training mean:")
print(f"  test R2   = {mean_baseline_r2:.4f}   (by definition, this is 0 in-sample)")
print(f"  test RMSE = {np.sqrt(np.mean((y_test - y_train.mean()) ** 2)):.4f}")
print()


# ======================================================================================
# PART 1 - variance is the criterion, and the SSE identity
# ======================================================================================
print()
print("-" * 78)
print("PART 1 - VARIANCE AS THE CRITERION")
print("-" * 78)


def variance(values):
    """Population variance of a 1-D array. Zero for an empty or single-row input."""
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return 0.0
    return float(np.var(values))


print("The reduction criterion, on a small worked example:")
print()
parent = np.array([10.0, 12.0, 11.0, 50.0])
print(f"parent node y = {parent.tolist()}")
print(f"  mean   {parent.mean():.4f}")
print(f"  variance (population) {variance(parent):.4f}")
sse_parent = np.sum((parent - parent.mean()) ** 2)
print(f"  SSE = SUM (y - mean)^2 = {sse_parent:.4f}")
print(f"  MSE = SSE/n = {sse_parent / len(parent):.4f}   <- the criterion")
print()
print(f"Note that variance is reported with a 1/n and SSE without, so they differ")
print(f"by exactly the factor n = {len(parent)}. {sse_parent:.4f} / {len(parent)} "
      f"= {sse_parent / len(parent):.4f}.")
print()

# Split and show the reduction.
left = parent[parent <= 11.5]
right = parent[parent > 11.5]
print(f"candidate split at 11.5:")
print(f"  left  {left.tolist()}  mean {left.mean():.4f}  SSE {np.sum((left - left.mean()) ** 2):.4f}")
print(f"  right {right.tolist()}  mean {right.mean():.4f}  "
      f"SSE {np.sum((right - right.mean()) ** 2):.4f}")
sse_left = np.sum((left - left.mean()) ** 2)
sse_right = np.sum((right - right.mean()) ** 2)
reduction = sse_parent - sse_left - sse_right
print()
print(f"  reduction = SSE(parent) - SSE(left) - SSE(right)")
print(f"           = {sse_parent:.4f} - {sse_left:.4f} - {sse_right:.4f}")
print(f"           = {reduction:.4f}")
print()
print(f"This is exactly what the tree maximises. The outlier at 50 dominates the")
print(f"parent variance, which is why the greedy split isolates it first - a tree")
print(f"handles outliers by putting them in their own leaf rather than by being")
print(f"robust to them. Script 15 covers robust alternatives.")
print()


# ======================================================================================
# PART 2 - what a leaf predicts, and why predictions are a staircase
# ======================================================================================
print()
print("-" * 78)
print("PART 2 - LEAF PREDICTIONS AND THE STAIRCASE")
print("-" * 78)

# A 1-D problem makes the piecewise-constant prediction visible.
rng = np.random.default_rng(SEED)
curve_x = np.sort(rng.uniform(0, 10, 400))
true_function = lambda x: 0.5 * x ** 2 - 3 * x + np.sin(x) * 4  # noqa: E731
curve_y = true_function(curve_x) + rng.normal(0, 2.0, 400)

train_x, test_x, train_y, test_y = train_test_split(
    curve_x, curve_y, test_size=0.3, random_state=SEED
)

print(f"A 1-D problem: y = 0.5x^2 - 3x + 4sin(x) + noise, n=400")
print()
print(f"{'max_depth':>10} {'leaves':>7} {'test RMSE':>11} {'test R2':>9} "
      f"{'distinct predictions':>22}")
print("-" * 68)
staircase_results = {}
for depth in [1, 2, 3, 5, 8, None]:
    model = DecisionTreeRegressor(max_depth=depth, random_state=SEED).fit(
        train_x.reshape(-1, 1), train_y)
    predictions = model.predict(test_x.reshape(-1, 1))
    distinct = len(np.unique(predictions))
    staircase_results[depth] = {
        "rmse": float(np.sqrt(mean_squared_error(test_y, predictions))),
        "r2": float(r2_score(test_y, predictions)),
        "distinct": distinct,
        "leaves": model.get_n_leaves(),
    }
    shown = "None" if depth is None else str(depth)
    print(f"{shown:>10} {model.get_n_leaves():>7} "
          f"{staircase_results[depth]['rmse']:>11.4f} "
          f"{staircase_results[depth]['r2']:>9.4f} {distinct:>22}")
print("-" * 68)
print()
print("The last column is the point. A depth-1 tree returns 2 distinct values for")
print(f"{len(test_x)} test rows; a depth-3 tree returns "
      f"{staircase_results[3]['distinct']}. The prediction function is a STEP")
print("function, not a curve. Each leaf is a flat horizontal segment.")
print()
print("This is the single biggest behavioural difference from linear regression: a")
print("tree cannot represent 0.5x^2 as a single expression. It gets curvature only")
print("by adding segments, so accuracy on a smooth non-linear target scales with")
print("depth rather than with the number of parameters - you pay for it in splits.")
print("Script 30 will separate the cases where that is worth it from those where a")
print("linear model is simply the better tool.")
print()

# Show the staircase numerically on a sorted grid.
grid = np.linspace(0, 10, 1001).reshape(-1, 1)
deep = DecisionTreeRegressor(max_depth=4, random_state=SEED).fit(
    train_x.reshape(-1, 1), train_y)
step_predictions = deep.predict(grid)
change_points = np.where(np.diff(step_predictions) != 0)[0]
print(f"depth-4 tree over the grid: {len(np.unique(step_predictions))} flat segments")
print(f"  jumps at x = "
      f"{', '.join(f'{grid[i, 0]:.2f}' for i in change_points[:10])}...")
print(f"  leaf value range {step_predictions.min():.2f} to {step_predictions.max():.2f}")
print(f"  training target range {train_y.min():.2f} to {train_y.max():.2f}")
print()
print("The leaf values sit inside the target range by construction. The tree can")
print("never predict outside [min(y_train), max(y_train)], which is a real")
print("limitation for extrapolation and a safety property for bounded outputs.")
print()

# Compare against linear regression on the same problem.
linear = LinearRegression().fit(train_x.reshape(-1, 1), train_y)
linear_predictions = linear.predict(test_x.reshape(-1, 1))
linear_rmse = float(np.sqrt(mean_squared_error(test_y, linear_predictions)))
curve_tree_rmse = staircase_results[5]["rmse"]
print("On this curved target:")
print(f"  depth-5 tree   test RMSE {curve_tree_rmse:.4f}")
print(f"  linear model  test RMSE {linear_rmse:.4f}")
print()
print(f"The tree wins by {linear_rmse - curve_tree_rmse:.4f} RMSE even though it is")
print("restricted to horizontal segments and the true function is a smooth curve.")
print("That is the staircase compensating for itself: with 1-D input, 400 rows and")
print("depth 5 it gets 32 free parameters, so approximating a parabola costs little.")
print()
print("The number to hold onto is the depth-1 row above, which achieves almost")
print(f"nothing (RMSE {staircase_results[1]['rmse']:.4f} from a single threshold). The")
print("tree's accuracy is bought with splits, and each split buys a flat segment.")
print("Whether that trade pays depends entirely on how much of the signal is")
print("non-linear and axis-aligned, which script 30 quantifies against a linear")
print("baseline on data built to have both.")


# ======================================================================================
# PART 3 - the fitted tree, rendered
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - A FITTED REGRESSION TREE")
print("-" * 78)

model = DecisionTreeRegressor(max_depth=3, random_state=SEED).fit(X_train, y_train)
print("diabetes, depth 3:")
print()
print(export_text(model, feature_names=feature_names, decimals=3))
print("The leaves show a value, not a class. Compare the leaf values to the leaf")
print("sample counts: a leaf holding 3 rows will report a mean that is unstable, and")
print("you have no way of knowing that from the export alone. Always read n next to")
print("the value.")
print()

# Retrieve leaf statistics directly from the tree structure.
print("Leaf statistics pulled from model.tree_:")
print()
print(f"{'node':>5} {'n':>5} {'value':>9} {'variance':>10} {'share of rows':>15}")
print("-" * 52)
tree = model.tree_
for node in range(tree.node_count):
    if tree.children_left[node] == -1:
        share = tree.weighted_n_node_samples[node] / tree.weighted_n_node_samples[0]
        print(f"{node:>5} {int(tree.n_node_samples[node]):>5} {tree.value[node, 0, 0]:>9.3f} "
              f"{tree.impurity[node]:>10.4f} {share:>14.1%}")
print("-" * 52)
print()
print("Two practical consequences. The smallest leaf's value is a mean over a handful")
print("of rows, so it is noisy; and impurity at a leaf is exactly the quantity that")
print("a split there was trying to reduce, which is how you find the worst leaves")
print("for manual review.")
print()


# ======================================================================================
# PART 4 - regularisation, and what it costs
# ======================================================================================
print()
print("-" * 78)
print("PART 4 - REGULARISATION AND THE ACCURACY/TRADE-OFF CURVE")
print("-" * 78)

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

# One fit per (depth, metric), collected in a single pass so the table below cannot
# disagree with itself. Note the scorer: R2 and RMSE are BOTH computed per fold.
# Reporting sqrt(-mean(R2)) would be a different quantity and is a common mistake -
# averaging R2 and then taking a square root does not recover an RMSE.
depths = [1, 2, 3, 4, 5, 6, 8, 10, None]
proper_sweep = {}
print(f"{'max_depth':>10} {'leaves':>7} {'CV RMSE':>10} {'CV MAE':>9} {'CV R2':>9} "
      f"{'train RMSE':>12} {'train-test gap':>16}")
print("-" * 78)
for depth in depths:
    rmse_scores = -cross_val_score(
        DecisionTreeRegressor(max_depth=depth, random_state=SEED),
        X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)
    mae_scores = -cross_val_score(
        DecisionTreeRegressor(max_depth=depth, random_state=SEED),
        X_train, y_train, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1)
    r2_scores = cross_val_score(
        DecisionTreeRegressor(max_depth=depth, random_state=SEED),
        X_train, y_train, cv=cv, scoring="r2", n_jobs=1)
    fitted = DecisionTreeRegressor(max_depth=depth, random_state=SEED).fit(X_train, y_train)
    train_predictions = fitted.predict(X_train)
    train_rmse = float(np.sqrt(mean_squared_error(y_train, train_predictions)))
    proper_sweep[depth] = {
        "rmse": float(rmse_scores.mean()),
        "mae": float(mae_scores.mean()),
        "r2": float(r2_scores.mean()),
        "leaves": fitted.get_n_leaves(),
        "train_rmse": train_rmse,
        "train_r2": float(r2_score(y_train, train_predictions)),
    }
    shown = "None" if depth is None else str(depth)
    gap = proper_sweep[depth]["r2"] - proper_sweep[depth]["train_r2"]
    print(f"{shown:>10} {fitted.get_n_leaves():>7} {rmse_scores.mean():>10.4f} "
          f"{mae_scores.mean():>9.4f} {r2_scores.mean():>9.4f} {train_rmse:>12.4f} "
          f"{gap:>16.4f}")
print("-" * 78)
print()
print("Read the gap column. It is CV R2 minus TRAIN R2, so it is negative and its")
print("magnitude grows with depth: the unpruned tree memorises the training set")
print(f"({proper_sweep[None]['train_r2']:.4f} train R2) while CV sits at only")
print(f"{proper_sweep[None]['r2']:.4f}. A gap of {abs(proper_sweep[None]['r2'] - proper_sweep[None]['train_r2']):.4f} is")
print("the overfitting signal, and it is visible before you touch the test set.")
print()

best_depth = min(proper_sweep, key=lambda d: proper_sweep[d]["rmse"])
print(f"Best by CV RMSE: max_depth={best_depth} "
      f"(RMSE {proper_sweep[best_depth]['rmse']:.4f})")
print()
best_by_mae = min(proper_sweep, key=lambda d: proper_sweep[d]["mae"])
print("RMSE and MAE can disagree about the best depth, because RMSE punishes the")
print("rare large error a deep tree makes by fitting an outlier leaf, while MAE")
print("spreads the penalty evenly across rows.")
print(f"  best depth by RMSE: {best_depth}")
print(f"  best depth by MAE : {best_by_mae}")
if best_depth == best_by_mae:
    print("On this dataset they agree, so the choice is not forced. When they do")
    print("disagree, pick by whether one bad prediction is worse than many small")
    print("errors - a business decision, not a technical one.")
else:
    print(f"They disagree here: {best_depth} by RMSE, {best_by_mae} by MAE. Pick by")
    print("whether one bad prediction is worse than many small errors.")
print()


# ======================================================================================
# PART 5 - tuning the real hyperparameter grid
# ======================================================================================
print()
print("-" * 78)
print("PART 5 - A PROPER GRID SEARCH")
print("-" * 78)

# Note the criterion options. squared_error, friedman_mse and absolute_error are
# the three real choices; poisson was removed for regression in sklearn 1.4.
print("criterion options for DecisionTreeRegressor:")
print("  squared_error    SSE reduction - the default, same objective as the docstring")
print("  friedman_mse     speed-up using per-feature orderings; usually near-identical")
print("  absolute_error   median-ish, much slower, worse on smooth targets")
print()
criterion_scores = {}
for criterion in ["squared_error", "friedman_mse", "absolute_error"]:
    scores = cross_val_score(
        DecisionTreeRegressor(max_depth=4, criterion=criterion, random_state=SEED),
        X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)
    criterion_scores[criterion] = float(-scores.mean())
    print(f"  {criterion:<18} CV RMSE {-scores.mean():.4f}")
print()
print("These are close enough that the choice is not worth a grid dimension.")
print()

grid_params = {
    "max_depth": [2, 3, 4, 5, 6, 8, None],
    "min_samples_leaf": [1, 5, 10, 20, 40],
    "min_samples_split": [2, 10, 20, 40],
    "max_features": [None, "sqrt", 0.5],
}
search = GridSearchCV(
    DecisionTreeRegressor(random_state=SEED),
    grid_params, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1,
)
search.fit(X_train, y_train)
print(f"GridSearchCV over {len(grid_params)} dimensions, "
      f"{int(np.prod([len(v) for v in grid_params.values()])):,} combinations:")
print(f"  best params  {search.best_params_}")
print(f"  best CV RMSE {-search.best_score_:.4f}")
print()
results = pd.DataFrame(search.cv_results_)[
    ["param_max_depth", "param_min_samples_leaf", "param_min_samples_split",
     "param_max_features", "mean_test_score", "rank_test_score"]].copy()
results.columns = ["max_depth", "min_leaf", "min_split", "max_features",
                   "CV RMSE", "rank"]
results["CV RMSE"] = -results["CV RMSE"]
results = results.sort_values("rank").head(10)
print("Top 10 configurations:")
print()
print(results.to_string(index=False))
print()
print("Read the top rows as a plateau rather than a peak: many configurations tie,")
print("and among tied ones you should take the SMALLEST tree, which is easier to")
print("explain and faster to predict. This is the 1-SE rule from the linear")
print("regression track, and it applies to trees for the same reason.")
print()


# ======================================================================================
# PART 6 - the tuned model on the held-out test set
# ======================================================================================
print()
print("-" * 78)
print("PART 6 - HELD-OUT EVALUATION")
print("-" * 78)

final = search.best_estimator_
test_predictions = final.predict(X_test)

print(f"Chosen: {search.best_params_}")
print(f"Tree: depth {final.get_depth()}, {final.get_n_leaves()} leaves, "
      f"{final.tree_.node_count} nodes")
print()
print(f"{'model':<28} {'R2':>9} {'RMSE':>9} {'MAE':>9}")
print("-" * 60)
comparison = {
    "constant (train mean)": np.full_like(y_test, y_train.mean()),
    "linear regression": LinearRegression().fit(X_train, y_train).predict(X_test),
    "tuned regression tree": test_predictions,
}
for name, predictions in comparison.items():
    print(f"{name:<28} {r2_score(y_test, predictions):>9.4f} "
          f"{np.sqrt(mean_squared_error(y_test, predictions)):>9.4f} "
          f"{mean_absolute_error(y_test, predictions):>9.4f}")
print("-" * 60)
tree_only = comparison["tuned regression tree"]
linear_only = comparison["linear regression"]
tree_rmse = float(np.sqrt(mean_squared_error(y_test, tree_only)))
linear_rmse_test = float(np.sqrt(mean_squared_error(y_test, linear_only)))
verdict = "beats" if tree_rmse < linear_rmse_test else "loses to"
print()
print(f"The tree {verdict} linear regression: {tree_rmse:.4f} vs {linear_rmse_test:.4f} RMSE, "
      f"{r2_score(y_test, tree_only):.4f} vs {r2_score(y_test, linear_only):.4f} R2.")
print()
print("That is the honest answer for this target, and the pattern in the depth sweep")
print("explains it. Every tree here is badly overfit at any useful depth, and CV R2")
print(f"peaks at {max(v['r2'] for v in proper_sweep.values()):.4f}. Ten physiological")
print("measurements map to disease progression close to linearly, so the signal that")
print("matters is a smooth trend the staircase cannot express, and the curvature the")
print("tree does add is noise. Trees win when the signal is non-linear,")
print("axis-aligned, and carried by interactions - script 30 makes that case on data")
print("built to have them, and it is the reason to check the baseline first.")
print()

residuals = y_test - test_predictions
print("Residual diagnostics:")
print(f"  mean     {residuals.mean():+.4f}   (near zero = unbiased)")
print(f"  std      {residuals.std():.4f}")
print(f"  skew     {pd.Series(residuals).skew():+.4f}")
print(f"  R2 vs prediction {np.corrcoef(np.abs(residuals), test_predictions)[0, 1]:+.4f}")
print(f"  target sd {y_test.std():.4f}")
print()
print("Residual sd against target sd is the practical read: the tree explains")
print(f"{1 - residuals.std() / y_test.std():.1%} of the spread on the held-out set.")
print()


# ======================================================================================
# Figures
# ======================================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
axes = axes.ravel()

# --- 1: the staircase ----------------------------------------------------------------
ax = axes[0]
order = np.argsort(test_x)
ax.plot(test_x[order], test_y[order], "o", color="#868E96", alpha=0.5, markersize=4,
       label="test data")
ax.plot(grid.ravel(), step_predictions, color="#4C6EF5", linewidth=2.5,
        label="depth-4 tree prediction")
ax.plot(np.linspace(0, 10, 200), true_function(np.linspace(0, 10, 200)), "--",
        color="#E03131", linewidth=2, label="true function")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("A regression tree predicts a staircase, not a curve", fontsize=11,
             fontweight="bold")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- 2: depth against error ----------------------------------------------------------
ax = axes[1]
shown_depths = list(proper_sweep)
positions = np.arange(len(shown_depths))
ax.plot(positions, [proper_sweep[d]["rmse"] for d in shown_depths], "o-",
        color="#4C6EF5", linewidth=2.2, label="CV RMSE")
ax.plot(positions, [proper_sweep[d]["mae"] for d in shown_depths], "s-",
        color="#0CA678", linewidth=2.2, label="CV MAE")
ax.set_xticks(positions)
ax.set_xticklabels(["None" if d is None else str(d) for d in shown_depths])
ax.set_xlabel("max_depth")
ax.set_ylabel("error on the target scale")
ax.set_title("RMSE and MAE can disagree about the best depth", fontsize=11,
             fontweight="bold")
ax.legend(fontsize=9, frameon=False)
ax.grid(alpha=0.25)

# --- 3: leaf values against leaf size ----------------------------------------------
ax = axes[2]
leaves = [node for node in range(tree.node_count) if tree.children_left[node] == -1]
leaf_counts = np.array([tree.n_node_samples[n] for n in leaves], dtype=float)
leaf_values = np.array([tree.value[n, 0, 0] for n in leaves])
ax.scatter(leaf_counts, leaf_values, s=140, color="#7048E8", edgecolors="white",
           linewidths=1.2)
for count, value in zip(leaf_counts, leaf_values):
    ax.annotate(f"{value:.1f}", xy=(count, value), xytext=(5, 5),
                textcoords="offset points", fontsize=8, color="#212529")
ax.set_xlabel("training rows in the leaf")
ax.set_ylabel("leaf prediction (mean of y)")
ax.set_title("Small leaves give unstable values - read n beside every value",
             fontsize=11, fontweight="bold")
ax.grid(alpha=0.25)

# --- 4: predicted vs actual ----------------------------------------------------------
ax = axes[3]
ax.scatter(y_test, test_predictions, s=26, alpha=0.6, color="#4C6EF5",
           edgecolors="none")
limit = max(y_test.max(), test_predictions.max()) * 1.03
ax.plot([0, limit], [0, limit], "--", color="#E03131", linewidth=2,
        label="perfect prediction")
ax.set_xlim(0, limit)
ax.set_ylim(0, limit)
ax.set_xlabel("actual")
ax.set_ylabel("predicted")
ax.set_title(f"Tuned tree, test R2 = {r2_score(y_test, test_predictions):.3f}",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

fig.suptitle("04 - Regression Trees", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


# ======================================================================================
# Report
# ======================================================================================
print()
print("=" * 78)
print("REPORT")
print("=" * 78)
print(f"""
DATASET          diabetes, {diabetes_X.shape[0]} rows x {diabetes_X.shape[1]} features,
                target mean {diabetes_y.mean():.1f} sd {diabetes_y.std():.1f}

CRITERION
  classification  Gini / entropy - impurity of class proportions
  regression      variance of targets = SSE/n, and the tree maximises
                  SSE(parent) - SSE(left) - SSE(right).  mse/rmse/friedman_mse are
                  the same objective up to a monotone rescaling.

LEAF PREDICTION  the mean of the training targets in that leaf, so the model is
                piecewise CONSTANT.  Measured on the 1-D curve above:
                  depth 1 -> {staircase_results[1]['distinct']} distinct predictions
                  depth 3 -> {staircase_results[3]['distinct']} distinct predictions
                  depth 8 -> {staircase_results[8]['distinct']} distinct predictions
                and leaf values cannot leave [min(y_train), max(y_train)]

DEPTH SWEEP (5-fold CV, diabetes)
  {'max_depth':>10} {'CV RMSE':>10} {'CV R2':>9} {'CV MAE':>9}
  {'-' * 42}
{chr(10).join(f"  {('None' if d is None else str(d)):>10} {proper_sweep[d]['rmse']:>10.4f} {proper_sweep[d]['r2']:>9.4f} {proper_sweep[d]['mae']:>9.4f}" for d in proper_sweep)}
  best by RMSE {best_depth}, best by MAE {best_by_mae}{'' if best_depth == best_by_mae else '  <- the metrics DISAGREE'}
  they can disagree, because RMSE punishes the large error a deep tree makes by
  fitting an outlier leaf; choosing between them is a decision about error
  asymmetry, not about scores

CRITERION COMPARISON (depth 4, CV RMSE)
{chr(10).join(f"  {name:<20} {value:.4f}" for name, value in criterion_scores.items())}

GRID SEARCH
  best params {search.best_params_}
  best CV RMSE {-search.best_score_:.4f}

HELD-OUT TEST SET
  {'model':<26} {'R2':>8} {'RMSE':>8} {'MAE':>8}
  {'-' * 54}
{chr(10).join(f"  {name:<26} {r2_score(y_test, p):>8.4f} {np.sqrt(mean_squared_error(y_test, p)):>8.4f} {mean_absolute_error(y_test, p):>8.4f}" for name, p in comparison.items())}
  constant baseline R2 {mean_baseline_r2:.4f} - the bar every model must clear

WHAT TO CARRY FORWARD
  - regression trees split to reduce VARIANCE, classification splits to reduce IMPURITY
  - a leaf predicts a mean, so predictions are a staircase and cannot extrapolate
  - RMSE and MAE agreed on depth here ({best_depth}), but they can disagree; when they
    do, the choice is about error asymmetry rather than about scores
  - squared_error / friedman_mse / mse are one objective, so do not grid over them
  - the top of a grid search is a plateau: take the smallest tied tree
  - a tree losing to linear regression on a smooth target is the correct result
""")
print("=" * 78)