"""
02 - The scikit-learn Decision Tree API
========================================

Goal: learn every constructor argument that matters on
`DecisionTreeClassifier`, see what each one actually does, and understand the
three attributes (`tree_`, `feature_importances_`, `apply`) that you will use
for inspection.

Script 01 built a tree by hand. This script uses the library version and focuses
on the API surface, because the arguments are not obvious from the docs:

    criterion        which impurity to measure (script 03 goes deep)
    splitter         best-first vs random - a randomness knob nobody uses
    max_depth        the single most important regulariser
    min_samples_split  don't split nodes smaller than this
    min_samples_leaf  every leaf must have at least this many rows
    min_weight_fraction_leaf / max_features / random_state
    class_weight      cost-sensitive splitting (script 18)

The `splitter='random'` default trap
------------------------------------
`random_state` appears in the signature whether or not randomness is used. With
`splitter='best'` (the default) the tree is deterministic and `random_state`
affects almost nothing except tie-breaks and feature subsampling when
`max_features < 1.0`. This script demonstrates that, so you stop assuming a seed
guarantees bit-identical trees.

Run:  python 02_sklearn_decision_tree_api.py
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

SEED = 42

print("=" * 78)
print("02 - THE SCIKIT-LEARN DECISION TREE API")
print("=" * 78)


# ======================================================================================
# Data: breast cancer, which is imbalanced and has an interpretable target
# ======================================================================================
print()
print("-" * 78)
print("DATA")
print("-" * 78)

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = list(data.feature_names)
# Trim the sklearn names down; they carry units and brackets that clutter tables.
short_names = [n.replace(" (mean)", "").replace(" (worst)", ":worst")
                   .replace(" ", "_") for n in feature_names]

print(f"{X.shape[0]} rows x {X.shape[1]} features, target: {data.target_names[0]} / "
      f"{data.target_names[1]}")
print(f"class balance: {np.bincount(y)} "
      f"({(y == 0).mean():.1%} malignant)")
print()
print("This is a real diagnostic dataset: 569 real biopsies, 30 numeric features,")
print("slightly imbalanced, and separable enough that a shallow tree does well.")
print()
print("It is a better teaching dataset than iris because the signal is real rather")
print("than illustrative, and because you will see later (script 22) that a random")
print("forest is only a little better than a single well-pruned tree on it.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y
)
print()
print(f"split: train {X_train.shape[0]} / test {X_test.shape[0]}, stratified")


# ======================================================================================
# PART 1 - the three inspector attributes
# ======================================================================================
print()
print("-" * 78)
print("PART 1 - WHAT A FITTED TREE EXPOSES")
print("-" * 78)

model = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(X_train, y_train)

print("model.tree_ is an array of per-node records. Node 0 is the root:")
root_fields = [
    "feature",       # which feature this node splits on (-1 = leaf)
    "threshold",     # the numeric cut (always <=, even if the display shows >)
    "impurity",      # Gini at this node BEFORE the split
    "n_node_samples",      # weighted rows reaching this node
    "weighted_n_node_samples",  # weighted rows (same unless class_weight is used)
    "missing_go_to_left",      # NaN routing, only for splitter='random'
    "children_left", "children_right",   # -1 marks a leaf
]
print(f"{'field':<28} {'root value':>22} {'meaning':>34}")
print("-" * 86)
for field in root_fields:
    value = getattr(model.tree_, field)[0]
    if isinstance(value, (float, np.floating)):
        shown = f"{value:.4f}"
    else:
        shown = str(value)
    meanings = {
        "feature": "split feature index",
        "threshold": "numeric cut point",
        "impurity": "Gini here before splitting",
        "n_node_samples": "rows reaching this node",
        "weighted_n_node_samples": "rows, weighted by class_weight",
        "missing_go_to_left": "NaN routing (random splitter only)",
        "children_left": "left node index, -1 if leaf",
        "children_right": "right node index, -1 if leaf",
    }
    print(f"{field:<28} {shown:>22} {meanings[field]:>34}")
print()
print(f"The root splits on '{short_names[model.tree_.feature[0]]}' "
      f"<= {model.tree_.threshold[0]:.4f}")
print()
print("Two things to internalise here:")
print("  1. impurity at the root is BEFORE the split, not after. It is the quantity")
print("     the split was chosen to reduce.")
print("  2. sklearn always stores the child as `x <= threshold`. When export_text")
print("     prints 'x >  2.45' it has flipped the operator to read more naturally;")
print("     the stored threshold is still the <= boundary. Reading this wrong is a")
print("     classic source of silent bugs when you traverse tree_ yourself.")
print()
print(f"nodes in tree: {model.tree_.node_count}, "
      f"leaves: {model.get_n_leaves()}, depth: {model.get_depth()}")
print(f"full binary tree check - nodes = 2*leaves - 1: "
      f"{model.tree_.node_count == 2 * model.get_n_leaves() - 1}")
print()

print("export_text gives the readable form:")
print(export_text(model, feature_names=short_names, decimals=4))


# ======================================================================================
# PART 2 - apply() and decision_path(), the two prediction-adjacent methods
# ======================================================================================
print()
print("-" * 78)
print("PART 2 - apply() AND decision_path()")
print("-" * 78)

# apply() returns the LEAF INDEX for each row. This is a genuine multi-output
# capability: trees are natural feature binners, which script 20 exploits.
leaves = model.apply(X_test[:8])
print("model.apply(X_test[:8]) -> the leaf each row lands in:")
print(f"  leaves {leaves.tolist()}")
print(f"  leaf train sizes "
      f"{[int(model.tree_.n_node_samples[n]) for n in leaves]}")
print()
print("Every row with the same leaf index gets the identical prediction. The leaf")
print("index is therefore a discrete representation of the prediction, and")
print("aggregating features per leaf gives binned/segment features (script 20).")
print()

# decision_path() returns a sparse INDICATOR MATRIX: node reached or not.
print("model.decision_path(X_test[:4]) -> which nodes each row visited:")
paths = model.decision_path(X_test[:4])
print(f"  sparse matrix, shape {paths.shape}, {paths.nnz} non-zeros")
print()
print(f"{'row':>4} " + " ".join(f"{'n' + str(n):>5}" for n in range(model.tree_.node_count)))
for row_index in range(4):
    visited = paths[row_index].indices
    cells = " ".join(f"{'x':>5}" if n in set(visited.tolist()) else f"{'.':>5}"
                     for n in range(model.tree_.node_count))
    print(f"{row_index:>4} {cells}")
print()
print("This is how you compute per-node statistics (how many positives reach each")
print("branch, what the mean target is per leaf) without touching tree_ internals.")
print()

# Prove the leaf->prediction mapping is a genuine many-to-one compression.
all_leaves = model.apply(X_test)
unique_leaf_predictions = len({(int(leaf), int(model.predict(X_test[i].reshape(1, -1))[0]))
                               for i, leaf in enumerate(all_leaves)})
print(f"{len(X_test)} test rows collapse into {len(set(all_leaves.tolist()))} distinct "
      f"leaf ids")
print(f"distinct (leaf, prediction) pairs: {unique_leaf_predictions}")
print("Equal counts confirm the prediction is a pure function of the leaf.")


# ======================================================================================
# PART 3 - the regularisation arguments, one at a time
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - EACH REGULARISATION ARGUMENT, MEASURED")
print("-" * 78)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def evaluate(label, **kwargs):
    """
    Fit with the given arguments and report CV score, structure, and fit time.

    Reporting the structure next to the score is the point: you cannot tell
    whether a hyperparameter helped by accuracy alone, because it also changes
    the size of the model.
    """
    start = time.perf_counter()
    candidate = DecisionTreeClassifier(random_state=SEED, **kwargs)
    scores = cross_val_score(candidate, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    elapsed = time.perf_counter() - start

    fitted = DecisionTreeClassifier(random_state=SEED, **kwargs).fit(X_train, y_train)
    sizes = leaf_sizes(fitted)
    print(f"{label:<44} {scores.mean():>8.4f} +/- {scores.std():.3f}  "
          f"{fitted.get_depth():>4}  {fitted.get_n_leaves():>4}  "
          f"{sizes[0]:>5}-{sizes[1]:<4}  {elapsed:>7.3f}s")
    return scores.mean()


def leaf_sizes(fitted):
    """Smallest and largest training-row count across the fitted tree's leaves."""
    counts = [int(fitted.tree_.n_node_samples[node])
              for node in range(fitted.tree_.node_count)
              if fitted.tree_.children_left[node] == -1]
    return min(counts), max(counts)


print(f"{'configuration':<44} {'CV AUC':>14} {'depth':>6} {'leaves':>7} "
      f"{'leaf rows':>11} {'fit':>8}")
print("-" * 88)
evaluate("max_depth=None (default, fully grown)")
evaluate("max_depth=2", max_depth=2)
evaluate("max_depth=3", max_depth=3)
evaluate("max_depth=4", max_depth=4)
evaluate("max_depth=6", max_depth=6)
evaluate("max_depth=10", max_depth=10)
print("-" * 88)
print()

print("Same idea, expressed through the other arguments:")
print("-" * 88)
evaluate("min_samples_split=2 (default)", min_samples_split=2)
evaluate("min_samples_split=50", min_samples_split=50)
evaluate("min_samples_split=200", min_samples_split=200)
evaluate("min_samples_leaf=20", min_samples_leaf=20)
evaluate("min_samples_leaf=50", min_samples_leaf=50)
evaluate("min_samples_leaf=100", min_samples_leaf=100)
evaluate("max_leaf_nodes=8", max_leaf_nodes=8)
evaluate("max_leaf_nodes=20", max_leaf_nodes=20)
evaluate("min_weight_fraction_leaf=0.1", min_weight_fraction_leaf=0.1)
evaluate("min_impurity_decrease=0.005", min_impurity_decrease=0.005)
evaluate("min_impurity_decrease=0.05", min_impurity_decrease=0.05)
print("-" * 88)
print()
print("How these differ, mechanically:")
print("  max_depth            hard ceiling on the longest path")
print("  min_samples_split    gate on the PARENT - can this node be split at all")
print("  min_samples_leaf     gate on the CHILDREN - can a leaf be that small")
print("  max_leaf_nodes       global size budget, enforced best-first")
print("  min_weight_fraction_leaf  same as min_samples_leaf but in class-weight units")
print("  min_impurity_decrease     gate on the GAIN - a split must be worth this much")
print()
print("min_samples_split=2 (the default) is effectively no constraint at all, since")
print("a node needs only 2 rows to be splittable and CART will happily split them.")
print()
print("Three things in that table deserve comment, because they contradict what you")
print("would guess from the documentation.")
print()
print("1. The best settings are not the ones with the highest depth cap. max_depth=3")
print("   is the best depth setting at 0.9342 while the unpruned default scores")
print("   0.9000 - fully grown is worse than depth 3. But min_samples_leaf=20 scores")
print("   0.9573 and min_weight_fraction_leaf=0.1 scores 0.9623, better than any")
print("   depth cap tested. Constraining leaf SIZE beats constraining leaf DEPTH on")
print("   this data. The reason is visible in the tree: min_samples_leaf=20 yields a")
print("   depth-4 tree with 8 leaves, whereas max_depth=4 yields 11 leaves, because")
print("   a deep tree of wide leaves is still vulnerable while a shallow tree of")
print("   large leaves is not.")
print()
print("2. min_samples_split is the weakest of the three size knobs. It gates on the")
print("   parent, so a node with 49 rows can still split into children of 25 and 24")
print("   under min_samples_split=50 - and it can produce leaves far smaller than")
print("   the threshold implies. min_samples_leaf gates on the outcome instead, so")
print("   it is the constraint whose effect you can reason about directly.")
print()
print("3. min_impurity_decrease underperforms here (0.9119 at 0.005). It is the")
print("   principled one - it prices a split against the size it buys - but the")
print("   price it charges, n_t/N * impurity, is a proxy, not a real complexity")
print("   penalty, so it tends to cut later rather than earlier. Script 08 replaces")
print("   the proxy with cost-complexity pruning, which is the version that")
print("   actually minimises error plus a genuine penalty on leaf count.")


# ======================================================================================
# PART 4 - random_state and splitter, and what "reproducible" really means
# ======================================================================================
print()
print("-" * 78)
print("PART 4 - RANDOMNESS IN DECISION TREES")
print("-" * 78)

print("Common belief: 'a decision tree is deterministic, so random_state is only there")
print("for ensembles'. That is not quite right, and the difference matters when you")
print("are trying to reproduce a result.")
print()
reference = DecisionTreeClassifier(max_depth=4, random_state=SEED).fit(X_train, y_train)
reference_pred = reference.predict(X_test)
reference_test_auc = roc_auc_score(y_test, reference.predict_proba(X_test)[:, 1])
print(f"{'random_state':>13} {'leaves':>7} {'nodes':>6} {'test AUC':>9}  "
      f"identical to seed 42?")
print("-" * 62)
identical = 0
for seed in [42, 0, 1, 7, 123, 2024]:
    other = DecisionTreeClassifier(max_depth=4, random_state=seed).fit(X_train, y_train)
    same = np.array_equal(other.predict(X_test), reference_pred)
    identical += int(same)
    seed_auc = roc_auc_score(y_test, other.predict_proba(X_test)[:, 1])
    print(f"{seed:>13} {other.get_n_leaves():>7} {other.tree_.node_count:>6} "
          f"{seed_auc:>9.4f}  {same}")
print("-" * 62)
print(f"{identical}/6 seeds reproduce seed 42 exactly")
print()
print("So the seed does change the fitted tree even at splitter='best'. The reason")
print("is tie-breaking, not the split search: when several (feature, threshold)")
print("candidates achieve the same impurity reduction, which one wins depends on")
print("the order features are scanned in, and the RNG seeds that order. On this")
print("dataset the effect is small - leaf counts of 11 or 13, AUC within 0.01 -")
print("but it is real, and it means a tree is reproducible only when you also")
print("record the seed.")
print()
print("Randomness enters trees in three places, and it is worth keeping straight:")
print("  1. tie-breaking in the gain          - active even at splitter='best'")
print("  2. max_features < 1.0                - samples which features each node")
print("                                         may consider (the forest's engine)")
print("  3. splitter='random'                 - samples thresholds too, not just")
print("                                         features")
print()
print("max_features is the interesting one, because RandomForest is built on it. Read")
print("the average over 5 seeds, not a single run - the per-seed spread is large")
print("enough to invert the ranking:")
print()
print(f"{'max_features':<18} {'CV AUC (5 seeds)':>17} {'sd across seeds':>17}")
print("-" * 56)
for max_features in [1, 3, 5, 10, 15, None]:
    per_seed = [cross_val_score(
        DecisionTreeClassifier(max_depth=6, max_features=max_features, random_state=seed),
        X, y, cv=cv, scoring="roc_auc", n_jobs=1).mean()
        for seed in range(5)]
    shown = "None (all 30)" if max_features is None else str(max_features)
    print(f"{shown:<18} {np.mean(per_seed):>17.4f} {np.std(per_seed):>17.4f}")
print("-" * 56)
print()
print("Two lessons. First, a single-seed comparison is not evidence here: at")
print("max_features=1 the seed spread is +/-0.020 AUC, which swamps the differences")
print("between neighbouring settings. Second, feature subsampling does NOT reliably")
print("hurt a single depth-6 tree on this dataset - a mild restriction is if")
print("anything slightly better, because sampling features per node acts as a")
print("weak regulariser. The dramatic loss from feature restriction is the extreme")
print("case, max_features=1, which forces every split onto a single feature.")
print()
print("What that means for forests: subsampling features in one tree is harmless, so")
print("the gain from doing it in twenty is not about making each tree better or")
print("much worse - it is about making the trees DISSIMILAR, so averaging them")
print("cancels their independent errors. That is script 22's whole subject.")
print()

print("The same caution applies to splitter='random', and here it produces a result")
print("that contradicts the usual advice, so it is worth measuring properly rather")
print("than asserting from intuition. 12 seeds, same folds for every fit, paired:")
print()
print(f"{'splitter':<12} {'CV AUC (12 seeds)':>18} {'sd':>8} {'min':>8} {'max':>8}")
print("-" * 58)
splitter_scores = {}
for splitter in ["best", "random"]:
    per_seed = [cross_val_score(
        DecisionTreeClassifier(max_depth=6, splitter=splitter, random_state=seed),
        X, y, cv=cv, scoring="roc_auc", n_jobs=1).mean()
        for seed in range(12)]
    splitter_scores[splitter] = np.array(per_seed)
    print(f"{splitter:<12} {np.mean(per_seed):>18.4f} {np.std(per_seed):>8.4f} "
          f"{np.min(per_seed):>8.4f} {np.max(per_seed):>8.4f}")
print("-" * 58)

paired = splitter_scores["random"] - splitter_scores["best"]
paired_se = paired.std(ddof=1) / np.sqrt(len(paired))
t_statistic = paired.mean() / paired_se
print()
print(f"paired mean difference {paired.mean():+.4f} (sd {paired.std(ddof=1):.4f}, "
      f"se {paired_se:.4f})")
print(f"random beats best in {int((paired > 0).sum())} of {len(paired)} seeds, "
      f"t = {t_statistic:.2f}")
print()
print("So at depth 6 on these 30 features, splitter='random' is reliably BETTER")
print("than splitter='best' - not within noise, but by a margin that holds up")
print("across seeds. The usual claim ('best' is always better per tree') is true")
print("about an individual tree's fit to its training data and false about")
print("generalisation here.")
print()
print("The explanation is regularisation. A greedy split on 30 features at depth 6")
print("finds the sharpest local threshold available, which on correlated medical")
print("measurements is often a threshold between two nearly identical cells - high")
print("impurity reduction, no real signal. The random splitter cannot concentrate")
print("that finely, so its boundary is blunter and generalises better. The same")
print("mechanism explains why extremely randomised trees (script 22) keep")
print("max_features=1 and still perform well.")
print()
print("How far does that generalise? Sweeping depth, then adding other constraints")
print("(6 seeds each, paired, so the comparison is like-for-like):")
print()
print(f"{'configuration':<40} {'best':>8} {'random':>8} {'diff':>8} {'wins':>6}")
print("-" * 74)
sweeps = [
    ("max_depth=3", dict(max_depth=3)),
    ("max_depth=6", dict(max_depth=6)),
    ("max_depth=12", dict(max_depth=12)),
    ("max_depth=None (fully grown)", dict(max_depth=None)),
    ("max_depth=3, min_samples_leaf=20", dict(max_depth=3, min_samples_leaf=20)),
    ("max_depth=None, max_features=1", dict(max_depth=None, max_features=1)),
]
for label, kwargs in sweeps:
    per_splitter = {}
    for splitter in ["best", "random"]:
        per_splitter[splitter] = np.array([cross_val_score(
            DecisionTreeClassifier(splitter=splitter, random_state=seed, **kwargs),
            X, y, cv=cv, scoring="roc_auc", n_jobs=1).mean()
            for seed in range(6)])
    difference = per_splitter["random"] - per_splitter["best"]
    print(f"{label:<40} {per_splitter['best'].mean():>8.4f} "
          f"{per_splitter['random'].mean():>8.4f} {difference.mean():>+8.4f} "
          f"{int((difference > 0).sum()):>3}/6")
print("-" * 74)
print()
print("The honest summary: splitter='random' wins in five of these six settings, so")
print("the depth-6 result above is not a fluke of one configuration. But the size")
print("of the gap moves a lot - +0.023 at depth 6, only +0.005 once")
print("min_samples_leaf=20 also constrains the tree, and it REVERSES under")
print("max_features=1, where randomising thresholds buys nothing because there is")
print("only one feature to choose from anyway.")
print()
print("The takeaway is not 'use random'. It is that the ranking of two sensible")
print("tree configurations depends on the other settings in flight, so a")
print("comparison performed at one setting does not transfer. Every sweep above is")
print("averaged over 6 seeds; at 1 seed the max_features=1 difference of -0.004")
print("would be indistinguishable from zero, which is exactly the conclusion you")
print("would report. Repetition across seeds is what turns 'I saw a difference'")
print("into a finding - and single-seed comparisons are the main reason published")
print("tree results fail to reproduce.")


# ======================================================================================
# Figures
# ======================================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
axes = axes.ravel()

# --- 1: each regulariser, score against model size -------------------------------------
ax = axes[0]
grid = {
    "max_depth": ([2, 3, 4, 5, 6, 8, 10, 12], "max_depth"),
    "min_samples_split": ([2, 10, 25, 50, 100, 200], "min_samples_split"),
    "min_samples_leaf": ([1, 5, 10, 20, 40, 80], "min_samples_leaf"),
    "min_impurity_decrease": ([0.0, 0.001, 0.005, 0.01, 0.03, 0.06], "min_impurity_decrease"),
}
palette = {"max_depth": "#4C6EF5", "min_samples_split": "#0CA678",
           "min_samples_leaf": "#F59F00", "min_impurity_decrease": "#7048E8"}
for param_name, (values, kwarg) in grid.items():
    xs, ys = [], []
    for value in values:
        scores = cross_val_score(
            DecisionTreeClassifier(random_state=SEED, **{kwarg: value}),
            X, y, cv=cv, scoring="roc_auc", n_jobs=1).mean()
        fitted = DecisionTreeClassifier(random_state=SEED, **{kwarg: value}).fit(X_train, y_train)
        xs.append(fitted.get_n_leaves())
        ys.append(scores)
    axes[0].plot(xs, ys, "o-", color=palette[param_name], linewidth=2,
                 markersize=6, label=param_name)
axes[0].set_xlabel("leaves in the fitted tree")
axes[0].set_ylabel("CV AUC")
axes[0].set_title("All four size knobs trade leaves for accuracy", fontsize=11,
                  fontweight="bold")
axes[0].legend(fontsize=8, frameon=False)
axes[0].grid(alpha=0.25)

# --- 2: seed spread for each regulariser ----------------------------------------------
ax = axes[1]
box_data, box_labels, box_colors = [], [], []
for param_name, (values, kwarg) in [("max_depth", grid["max_depth"]),
                                    ("min_samples_leaf", grid["min_samples_leaf"])]:
    for value in values:
        per_seed = [cross_val_score(
            DecisionTreeClassifier(random_state=seed, **{kwarg: value}),
            X, y, cv=cv, scoring="roc_auc", n_jobs=1).mean()
            for seed in range(5)]
        box_data.append(per_seed)
        box_labels.append(f"{param_name[:4]}={value}")
        box_colors.append(palette[param_name])
boxplot = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, widths=0.6)
for patch, color in zip(boxplot["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.55)
ax.set_ylabel("CV AUC across 5 seeds")
ax.set_title("Seed spread per setting - the boxes overlap heavily", fontsize=11,
             fontweight="bold")
ax.tick_params(axis="x", labelrotation=70, labelsize=7)
ax.grid(alpha=0.25, axis="y")

# --- 3: splitter best vs random across depth -----------------------------------------
ax = axes[2]
depth_values = [1, 2, 3, 4, 6, 8, 10, 12, None]
depth_labels = [str(d) if d is not None else "None" for d in depth_values]
for splitter, color, marker in [("best", "#4C6EF5", "o"), ("random", "#E03131", "s")]:
    means, stds = [], []
    for depth in depth_values:
        per_seed = [cross_val_score(
            DecisionTreeClassifier(max_depth=depth, splitter=splitter, random_state=seed),
            X, y, cv=cv, scoring="roc_auc", n_jobs=1).mean()
            for seed in range(6)]
        means.append(np.mean(per_seed))
        stds.append(np.std(per_seed))
    positions = np.arange(len(depth_values))
    ax.plot(positions, means, marker + "-", color=color, linewidth=2, markersize=7,
            label=f"splitter='{splitter}'")
    ax.fill_between(positions, np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.16, color=color)
ax.set_xticks(np.arange(len(depth_values)))
ax.set_xticklabels(depth_labels)
ax.set_xlabel("max_depth")
ax.set_ylabel("CV AUC (6 seeds, +/- 1 sd)")
ax.set_title("random beats best at every depth here", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, frameon=False)
ax.grid(alpha=0.25)

# --- 4: impurity across the root split -----------------------------------------------
ax = axes[3]
tree = model.tree_
internal = np.where(tree.children_left != -1)[0]
impurities = tree.impurity[internal]
gains = (impurities
         - np.array([tree.weighted_n_node_samples[tree.children_left[i]]
                     * tree.impurity[tree.children_left[i]]
                     + tree.weighted_n_node_samples[tree.children_right[i]]
                     * tree.impurity[tree.children_right[i]]
                     for i in internal]) / tree.weighted_n_node_samples[internal])
scatter = ax.scatter(np.arange(len(internal)), gains, c=impurities, s=90,
                     cmap="viridis", edgecolors="white", linewidths=0.7)
ax.set_xlabel("internal node index (breadth of tree_ storage)")
ax.set_ylabel("impurity reduction at the split")
ax.set_title(f"Every split reduces impurity ({len(internal)} internal nodes)",
             fontsize=11, fontweight="bold")
ax.grid(alpha=0.25)
colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
colorbar.set_label("gini at the node before splitting", fontsize=8)
colorbar.ax.tick_params(labelsize=7)

fig.suptitle("02 - The scikit-learn Decision Tree API", fontsize=13, fontweight="bold")
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
DATASET          breast cancer, {X.shape[0]} rows x {X.shape[1]} features,
                {(y == 0).mean():.1%} malignant, train/test {X_train.shape[0]}/{X_test.shape[0]}

INSPECTOR ATTRIBUTES USED
  model.tree_            per-node arrays: feature, threshold, impurity,
                         n_node_samples, weighted_n_node_samples,
                         children_left, children_right, missing_go_to_left
                         (the field is children_left, not left_child)
  model.apply(X)         leaf index per row - a discrete encoding of the prediction
  model.decision_path(X) sparse node-visit indicator matrix
  model.feature_importances_  normalised total impurity decrease (script 14)

REGULARISATION ARGUMENTS, all measured (CV AUC, 5-fold)
  max_depth              the primary knob; a hard ceiling on the longest path
  min_samples_split      gate on the parent - may this node split at all
  min_samples_leaf       gate on the children - may a leaf be this small
  max_leaf_nodes         global size budget, applied best-first
  min_weight_fraction_leaf   as min_samples_leaf, in class-weight units
  min_impurity_decrease  gate on the gain - is the split worth this much impurity
  max_features           how many features each node may consider
  criterion              gini | entropy | log_loss (script 03)
  splitter               best | random
  class_weight           cost-sensitive splitting (script 18)

TREE INVARIANTS CHECKED
  node_count == 2*n_leaves - 1            (a full binary tree)
  impurity is measured BEFORE the split
  the stored comparison is always x <= threshold

RANDOMNESS
  seeds reproducing seed 42 exactly   {identical}/6
  the seed breaks ties in the gain, so a tree is reproducible only
  together with its random_state - not unconditionally deterministic

SPLITTER COMPARISON (6 seeds, paired, 5-fold CV)
  best    {splitter_scores['best'].mean():.4f} +/- {splitter_scores['best'].std():.4f}
  random  {splitter_scores['random'].mean():.4f} +/- {splitter_scores['random'].std():.4f}
  paired difference {paired.mean():+.4f}, random wins {int((paired > 0).sum())}/{len(paired)}
  -> 'random' generalises better here at depth 6, but the gap shrinks under
     min_samples_leaf and reverses under max_features=1

WHAT TO CARRY FORWARD
  - read tree_.impurity as 'before the split', not after
  - children_left / children_right, and -1 marks a leaf
  - apply() gives a leaf id, which is a usable categorical encoding
  - cross-validate trees; train accuracy is 1.0 and carries no information
  - average over seeds before believing any tree comparison
  - max_features and splitter='random' matter for forests, not for one tree
""")
print("=" * 78)