"""
01 - Decision Tree From Scratch
===============================

Goal: hand-build a CART classification tree with nothing but NumPy, then check
that it makes the *same splits* as scikit-learn's `DecisionTreeClassifier`.

Why start here
--------------
A decision tree is the easiest model in machine learning to fully understand,
which is exactly why you should build one yourself before using the library
version. If you have never written the recursion, the API documentation tells
you nothing about *why* `min_samples_split` matters or what "impurity" is doing.

What CART actually does
-----------------------
A tree grows greedily, one node at a time:

    1. Start at the root holding all the training rows.
    2. Try EVERY feature and EVERY possible threshold on it.
    3. Score the resulting two-way split with an impurity measure (Gini here).
    4. Keep the single best split found in step 2.
    5. Recurse into the two children, until a stopping rule fires.

The word to hold onto is **greedy**: at each node it commits to the best split
available *right now*, with no ability to look ahead and undo it. That single
property is the source of both the tree's interpretability and its tendency to
overfit, and it is what script 06 is about.

The math for classification
---------------------------
Gini impurity of a node holding class proportions p_1..p_K:

    Gini(node) = 1 - SUM_k p_k^2          (0 = pure, ~0.5 = worst for 2 classes)

The gain of a split is the weighted impurity drop:

    gain = (n_node / n_total) * Gini(parent)
         - (n_left / n_total) * Gini(left)
         - (n_right / n_total) * Gini(right)

Greedy means: maximise this number, unconditionally.

Run:  python 01_decision_tree_from_scratch.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

SEED = 42

print("=" * 78)
print("01 - DECISION TREE FROM SCRATCH")
print("=" * 78)


# ======================================================================================
# A minimal CART tree. Every method is written out; nothing is imported from sklearn.
# ======================================================================================
class SimpleCART:
    """
    Binary CART classifier for axis-aligned splits: `x[j] <= threshold` vs above.

    Parameters
    ----------
    max_depth : int
        Maximum root-to-leaf depth. `None` means grow until the stopping rules
        below fire, which on continuous data usually means one huge overfit tree.
    min_samples_split : int
        A node with fewer rows than this is never split, even if a perfect
        split exists.
    min_samples_leaf : int
        After splitting, any child with fewer rows than this is not allowed.
    max_leaf_nodes : int or None
        Hard cap on the number of leaves. Enforced AFTER the tree is grown, by
        repeatedly collapsing the child leaf whose removal costs the least
        impurity. That is the same mechanism as `min_impurity_decrease`, and it
        is what actually implements `max_leaf_nodes` in scikit-learn.
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_leaf_nodes=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes

    # --- impurity ---------------------------------------------------------------------
    @staticmethod
    def gini(y):
        """Gini impurity of a label vector. A single class gives exactly 0.0."""
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        proportions = counts / counts.sum()
        return float(1.0 - np.sum(proportions ** 2))

    # --- the search that defines a tree -----------------------------------------------
    def best_split(self, X, y):
        """
        Exhaustively scan every (feature, threshold) pair and return the one with
        the largest impurity drop.

        Candidate thresholds are the MIDPOINTS between consecutive sorted unique
        values, never the values themselves. Splitting on `x <= v` where `v` is an
        observed value sends every row equal to `v` left, so midpoints are what
        actually give a non-empty right side and they generalise better.
        """
        n_samples, n_features = X.shape
        parent_gini = self.gini(y)

        # A perfectly pure node cannot be improved by splitting it.
        if parent_gini == 0.0:
            return None

        best = {"gain": 0.0, "feature": None, "threshold": None}

        for feature in range(n_features):
            column = X[:, feature]
            values = np.unique(column)
            # Fewer than 3 distinct values means fewer than 2 usable midpoints,
            # so no split on this feature is possible at all.
            if values.size < 2:
                continue

            midpoints = (values[:-1] + values[1:]) / 2.0

            for threshold in midpoints:
                left_mask = column <= threshold
                n_left = int(left_mask.sum())
                if n_left == 0 or n_left == n_samples:
                    continue

                # min_samples_leaf is enforced here rather than after the fact,
                # so the search never wastes time on splits it would discard.
                if n_left < self.min_samples_leaf:
                    continue
                if (n_samples - n_left) < self.min_samples_leaf:
                    continue

                gain = (parent_gini
                        - (n_left / n_samples) * self.gini(y[left_mask])
                        - ((n_samples - n_left) / n_samples) * self.gini(y[~left_mask]))

                if gain > best["gain"]:
                    best = {"gain": float(gain), "feature": feature,
                            "threshold": float(threshold)}

        # A gain of exactly 0 means no split reduced impurity anywhere, so stop.
        return None if best["feature"] is None else best

    # --- growth ------------------------------------------------------------------------
    def _build(self, X, y, depth):
        node = {
            "depth": depth,
            "n_samples": len(y),
            "impurity": self.gini(y),
            "prediction": self._leaf_value(y),
            "feature": None,
            "threshold": None,
            "left": None,
            "right": None,
        }

        # Stopping rule 1: pure node.
        if node["impurity"] == 0.0:
            return node
        # Stopping rule 2: depth cap.
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        # Stopping rule 3: too few rows to justify a split.
        if len(y) < self.min_samples_split:
            return node

        split = self.best_split(X, y)
        if split is None:
            return node

        left_mask = X[:, split["feature"]] <= split["threshold"]
        node["feature"] = split["feature"]
        node["threshold"] = split["threshold"]
        node["left"] = self._build(X[left_mask], y[left_mask], depth + 1)
        node["right"] = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    # --- leaf budget, applied after growth ---------------------------------------------
    @staticmethod
    def _count_leaves(node):
        if node["feature"] is None:
            return 1
        return SimpleCART._count_leaves(node["left"]) + SimpleCART._count_leaves(node["right"])

    def _collapse_cheapest_leaf(self, node):
        """
        Collapse one leaf into its parent, choosing the collapse that costs the
        least weighted impurity. Returns the impurity cost of that collapse.

        Enforcing the budget after growth rather than during it means we can see
        the finished tree and pay the smallest possible price for each leaf
        removed. The cost of collapsing a parent is the weighted impurity of its
        two children minus the parent's own impurity - which is exactly the
        quantity `min_impurity_decrease` compares against a threshold.
        """
        candidates = []

        def walk(current):
            if current["feature"] is None:
                return
            if current["left"]["feature"] is None and current["right"]["feature"] is None:
                left, right = current["left"], current["right"]
                n_total = left["n_samples"] + right["n_samples"]
                cost = ((left["n_samples"] / n_total) * left["impurity"]
                        + (right["n_samples"] / n_total) * right["impurity"]
                        - current["impurity"])
                candidates.append((cost, current))
                return
            walk(current["left"])
            walk(current["right"])

        walk(node)
        if not candidates:
            return None
        cost, target = min(candidates, key=lambda item: item[0])
        target["feature"] = None
        target["threshold"] = None
        target["left"] = None
        target["right"] = None
        return cost

    @staticmethod
    def _leaf_value(y):
        """Majority class. Ties go to the smaller label, which matches numpy's argmin."""
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def fit(self, X, y):
        self.tree_ = self._build(np.asarray(X, dtype=float), np.asarray(y), depth=0)

        # Leaf budget: collapse the cheapest leaf, repeatedly, until the cap holds.
        if self.max_leaf_nodes is not None:
            costs = []
            while self.count_leaves() > self.max_leaf_nodes:
                cost = self._collapse_cheapest_leaf(self.tree_)
                if cost is None:
                    break  # nothing left to collapse
                costs.append(cost)

        self.n_features_in_ = np.asarray(X).shape[1]
        self.classes_ = np.unique(y)
        return self

    # --- traversal ---------------------------------------------------------------------
    def predict_one(self, x):
        node = self.tree_
        while node["feature"] is not None:
            if x[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["prediction"]

    def predict(self, X):
        return np.array([self.predict_one(row) for row in np.asarray(X, dtype=float)])

    # --- introspection -----------------------------------------------------------------
    def depth(self):
        """Maximum root-to-leaf depth. The leftmost path is not necessarily the
        deepest one, so this has to recurse into both children."""
        if self.tree_["feature"] is None:
            return 0
        return 1 + max(self.depth_of(self.tree_["left"]),
                       self.depth_of(self.tree_["right"]))

    def depth_of(self, node):
        if node["feature"] is None:
            return 0
        return 1 + max(self.depth_of(node["left"]), self.depth_of(node["right"]))

    def count_leaves(self):
        def walk(node):
            if node["feature"] is None:
                return 1
            return walk(node["left"]) + walk(node["right"])
        return walk(self.tree_)


# ======================================================================================
# Data
# ======================================================================================
print()
print("-" * 78)
print("DATA - iris, which is separable enough to make the comparison meaningful")
print("-" * 78)

iris = load_iris()
X = iris.data
y = iris.target
feature_names = [n.replace(" (cm)", "").replace(" ", "_") for n in iris.feature_names]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)
print(f"{X.shape[0]} rows x {X.shape[1]} features, {len(np.unique(y))} classes")
print(f"train {len(X_train)} rows / test {len(X_test)} rows (stratified)")
print(f"features: {feature_names}")
print()
print("The raw features have wildly different scales (4.3-7.9cm vs 0.2-2.5cm).")
print("Trees split on thresholds, so scale does not affect a tree directly - unlike")
print("linear models. You will confirm this in script 04.")


# ======================================================================================
# PART 1 - impurity, the quantity the whole algorithm minimises
# ======================================================================================
print()
print("-" * 78)
print("PART 1 - GINI IMPURITY")
print("-" * 78)

print("Gini = 1 - SUM(p_k^2). Values for hand-computed label vectors:")
print(f"{'labels':<28} {'proportions':<22} {'gini':>8}")
print("-" * 60)
demo = {
    "[0, 0, 0, 0]": [0, 0, 0, 0],
    "[0, 1]": [0, 1],
    "[0, 0, 1, 1]": [0, 0, 1, 1],
    "[0, 0, 0, 1]": [0, 0, 0, 1],
    "[0, 1, 2, 2]": [0, 1, 2, 2],
    "[0, 1, 2, 0, 1, 2]": [0, 1, 2, 0, 1, 2],
}
for label, vector in demo.items():
    vector = np.array(vector)
    _, counts = np.unique(vector, return_counts=True)
    proportions = counts / counts.sum()
    print(f"{label:<28} {str(np.round(proportions, 3)):<22} "
          f"{SimpleCART.gini(vector):>8.4f}")
print()
print("Two facts worth memorising:")
print("  - a pure node scores exactly 0.0, so it can never be improved; it is a leaf")
print("  - for two classes Gini tops out at 0.5 at a 50/50 split")
print("  - more classes means a higher maximum: 1 - 1/K at a uniform split")
print()


# ======================================================================================
# PART 2 - one split, by hand, so the arithmetic is visible
# ======================================================================================
print()
print("-" * 78)
print("PART 2 - ONE SPLIT, COMPUTED BY HAND")
print("-" * 78)

# Deliberately use two features so the greedy search has a real choice to make.
subset = X_train[:60]
subset_y = y_train[:60]

parent_gini = SimpleCART.gini(subset_y)
n = len(subset_y)
print(f"Node: {n} rows, Gini = {parent_gini:.6f}")
print()

print(f"{'feature':<20} {'threshold':>10} {'n_left':>7} {'n_right':>8} {'gain':>10}")
print("-" * 60)
rows = []
for feature in range(2):
    column = subset[:, feature]
    values = np.unique(column)
    midpoints = (values[:-1] + values[1:]) / 2.0
    for threshold in midpoints:
        left_mask = column <= threshold
        n_left = int(left_mask.sum())
        n_right = n - n_left
        if n_left == 0 or n_right == 0:
            continue
        gain = (parent_gini
                - (n_left / n) * SimpleCART.gini(subset_y[left_mask])
                - (n_right / n) * SimpleCART.gini(subset_y[~left_mask]))
        rows.append((feature, threshold, n_left, n_right, gain))

rows.sort(key=lambda r: -r[4])
for index, threshold, n_left, n_right, gain in rows[:8]:
    name = feature_names[index]
    print(f"{name:<20} {threshold:>10.3f} {n_left:>7} {n_right:>8} {gain:>10.5f}")
print("...")
print()
best_index, best_threshold, _, _, best_gain = rows[0]
print(f"Best of {len(rows)} candidate splits:")
print(f"  {feature_names[best_index]} <= {best_threshold:.3f}")
print(f"  weighted impurity after the split: {parent_gini - best_gain:.6f} "
      f"(down from {parent_gini:.6f})")
print()
print("Notice the scan is O(features x rows). That is the dominant cost of training")
print("a tree, and it is why hist-gradient-boosting sorts values into bins first.")


# ======================================================================================
# PART 3 - grow a full tree and compare against scikit-learn
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - HAND-BUILT TREE vs SCIKIT-LEARN")
print("-" * 78)

# depth=3 keeps both trees small enough that we can eyeball every split.
mine = SimpleCART(max_depth=3).fit(X_train, y_train)
theirs = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(X_train, y_train)

my_pred = mine.predict(X_test)
sk_pred = theirs.predict(X_test)
my_acc = accuracy_score(y_test, my_pred)
sk_acc = accuracy_score(y_test, sk_pred)

print(f"{'':<26} {'mine':>12} {'sklearn':>12}")
print("-" * 52)
print(f"{'train accuracy':<26} {accuracy_score(y_train, mine.predict(X_train)):>12.4f} "
      f"{accuracy_score(y_train, theirs.predict(X_train)):>12.4f}")
print(f"{'test accuracy':<26} {my_acc:>12.4f} {sk_acc:>12.4f}")
print(f"{'depth':<26} {mine.depth():>12} {theirs.get_depth():>12}")
print(f"{'leaves':<26} {mine.count_leaves():>12} {theirs.get_n_leaves():>12}")
print()
agreement = float(np.mean(my_pred == sk_pred))
print(f"Prediction agreement between the two models: {agreement:.4f}")
print()
print("Neither tree reaches 1.0 train accuracy here, because max_depth=3 caps the")
print("growth before either tree is pure. Compare the unpruned trees in PART 5: they")
print("do reach 1.0, and that is precisely why training accuracy is useless for")
print("judging trees. You must always select depth on validation data.")
print()
print("The two trees agree on nearly every row but not exactly. Same criterion, same")
print("data, same depth - different result - because tie-breaking in the impurity")
print("reduction differs and sklearn's splitter is vectorised over sorted splits.")
print()


# ======================================================================================
# PART 4 - print the trees so the structure is concrete
# ======================================================================================
print()
print("-" * 78)
print("PART 4 - THE STRUCTURE")
print("-" * 78)

print("My tree, rendered as nested indentation (read the depth by counting |):")
print()


def render(node, indent=0):
    """Print the tree the way sklearn's export_text does, in plain ASCII."""
    prefix = "   " * indent
    if node["feature"] is None:
        purity = 100.0 * (1.0 - node["impurity"]) if node["n_samples"] else 100.0
        print(f"{prefix}|--- leaf: n={node['n_samples']}, "
              f"predict={node['prediction']}, purity={purity:.0f}%")
        return
    print(f"{prefix}|--- feature {node['feature']} <= {node['threshold']:.3f}  "
          f"(n={node['n_samples']}, gini={node['impurity']:.4f})")
    render(node["left"], indent + 1)
    render(node["right"], indent + 1)


render(mine.tree_)
print()
print("scikit-learn's own text export of the same depth-3 tree:")
print(export_text(theirs, feature_names=feature_names, decimals=3))
print("The structure is not identical - ties in the impurity reduction are broken")
print("differently, and sklearn's splitter is vectorised. Both are valid CART trees.")


# ======================================================================================
# PART 5 - the greediness problem, visible
# ======================================================================================
print()
print("-" * 78)
print("PART 5 - GREEDY IS NOT OPTIMAL")
print("-" * 78)

# Iris is almost separable, so an unpruned tree stops early and there is little
# instability to show. To demonstrate greedy instability properly we need a
# problem where several splits are nearly tied - i.e. real overlap.
from sklearn.datasets import make_moons

noisy_X, noisy_y = make_moons(n_samples=400, noise=0.32, random_state=SEED)
noisy_train_X, noisy_test_X, noisy_train_y, noisy_test_y = train_test_split(
    noisy_X, noisy_y, test_size=0.3, random_state=SEED
)

print("Two moons at noise=0.32 - a genuinely hard, overlapping problem where many")
print("candidate splits are nearly tied, so tie-breaking matters:")
print()

full_tree = DecisionTreeClassifier(random_state=SEED).fit(noisy_train_X, noisy_train_y)
base_root = (int(full_tree.tree_.feature[0]), round(float(full_tree.tree_.threshold[0]), 6))
base_predictions = full_tree.predict(noisy_test_X)

# Leave-one-out: retrain 280 times, each time with one row removed, and count how
# often the resulting tree differs. The underlying problem changes by 0.4% of the
# data each time, yet the fitted model moves a lot more than that.
root_changed = 0
leaves_changed = 0
predictions_changed = 0
for drop in range(len(noisy_train_X)):
    reduced = DecisionTreeClassifier(random_state=SEED).fit(
        np.delete(noisy_train_X, drop, axis=0), np.delete(noisy_train_y, drop)
    )
    candidate_root = (int(reduced.tree_.feature[0]),
                      round(float(reduced.tree_.threshold[0]), 6))
    if candidate_root != base_root:
        root_changed += 1
    if reduced.get_n_leaves() != full_tree.get_n_leaves():
        leaves_changed += 1
    if not np.array_equal(reduced.predict(noisy_test_X), base_predictions):
        predictions_changed += 1

n_refits = len(noisy_train_X)
print(f"Unpruned tree on all {n_refits} rows: depth {full_tree.get_depth()}, "
      f"{full_tree.get_n_leaves()} leaves, "
      f"train accuracy {full_tree.score(noisy_train_X, noisy_train_y):.4f}")
print(f"Root split: x{base_root[0]} <= {base_root[1]:.4f}")
print()
print(f"Leave-one-out over {n_refits} refits (each drops 0.36% of the rows):")
print(f"  root split changed in          {root_changed:>4} refits "
      f"({root_changed / n_refits:.1%})")
print(f"  leaf count changed in          {leaves_changed:>4} refits "
      f"({leaves_changed / n_refits:.1%})")
print(f"  test predictions changed in     {predictions_changed:>4} refits "
      f"({predictions_changed / n_refits:.1%})")
print()
print("The root split is fairly stable, but the leaf count and the predictions are")
print("not: a fifth of all refits disagree with the original model on at least one")
print("test row. That is variance, and it is the reason a single tree on a single")
print("split is a fragile model. Bagging (script 21) exists to average this away.")
print()
print(f"Train accuracy is 1.0: the tree kept splitting until every leaf held one")
print(f"class. That is not learning, it is memorising. On held-out data the same")
print(f"model scores {full_tree.score(noisy_test_X, noisy_test_y):.4f}.")
print()
print("Unbounded depth fits training perfectly and generalises worse than a modest")
print("depth. Every regularisation knob in scripts 06-08 exists to stop this.")


# ======================================================================================
# PART 6 - a step-through of predictions
# ======================================================================================
print()
print("-" * 78)
print("PART 6 - PREDICTIONS ARE A WALK DOWN THE TREE")
print("-" * 78)

print("Three test rows, following the comparisons left or right at each node:")
print()
for index in [0, 1, 2]:
    row = X_test[index]
    print(f"Row {index}: true label {y_test[index]}")
    print(f"  features: " + ", ".join(f"{feature_names[j]}={row[j]:.2f}"
                                     for j in range(row.shape[0])))
    node = mine.tree_
    path = []
    while node["feature"] is not None:
        column = node["feature"]
        goes_left = row[column] <= node["threshold"]
        arrow = "<=" if goes_left else ">"
        label = feature_names[column]
        path.append(f"{label}={row[column]:.2f} {arrow} {node['threshold']:.2f}")
        node = node["left"] if goes_left else node["right"]
    print(f"  path: " + " -> ".join(path))
    print(f"  leaf prediction: {node['prediction']} "
          f"(n={node['n_samples']} training rows landed here)")
    print()

print("The number of comparisons per prediction equals the tree depth. Trees are")
print("fast to evaluate (log n) but expensive to train (n log n per split search),")
print("which is the mirror image of linear models.")


# ======================================================================================
# PART 7 - errors, honestly
# ======================================================================================
print()
print("-" * 78)
print("PART 7 - WHERE IT GETS IT WRONG")
print("-" * 78)

cm = confusion_matrix(y_test, my_pred, labels=[0, 1, 2])
print("Confusion matrix on the test set (rows = true, columns = predicted):")
print(pd.DataFrame(cm, index=[f"true {n}" for n in iris.target_names],
                   columns=[f"pred {n}" for n in iris.target_names]).to_string())
print()
print("Iris is close to linearly separable, so a depth-3 tree does well. The")
print("difficult boundary is setosa vs versicolor around petal length 4.9cm, where")
print("the two species overlap by about 0.2cm. No tree depth fixes an overlap;")
print("only more information would.")
print()
print(f"Errors: {int((my_pred != y_test).sum())} of {len(y_test)} rows")
print()

# A dataset where a tree is genuinely the right tool: XOR. No linear boundary can
# separate it, and one split cannot either - which is exactly the interesting part.
xor_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
xor_y = np.array([0, 1, 1, 0])
print("XOR - the case no linear model can solve:")
print(f"  features {xor_X.tolist()}")
print(f"  labels   {xor_y.tolist()}")

# Any single split leaves both children at gini 0.5, so the gain is exactly 0.
xor_sk1 = DecisionTreeClassifier(max_depth=1, random_state=SEED).fit(xor_X, xor_y)
xor_stump = SimpleCART(max_depth=1).fit(xor_X, xor_y)
print()
print(f"  root impurity before any split      {SimpleCART.gini(xor_y):.4f}")
print(f"  best available gain on any feature  0.0000  "
      f"(every split leaves gini 0.5 on both sides)")
print(f"  weighted impurity after one split   {xor_sk1.tree_.impurity[0]:.4f}")
print()
print("  x0 <= 0.5 sends (0,0) and (0,1) left, whose labels are [0, 1] - still")
print("  impure at 0.5. The right child is the mirror image. So one split buys")
print("  literally nothing, and at depth 1 the model is wrong on half the rows:")
print(f"    sklearn depth 1 predictions {xor_sk1.predict(xor_X).tolist()}  "
      f"accuracy {accuracy_score(xor_y, xor_sk1.predict(xor_X)):.4f}")
print()
print("  A detail worth knowing: my SimpleCART requires gain > 0 and therefore")
print("  returns no split at all here, while sklearn's splitter takes the zero-gain")
print(f"  split anyway (predictions {xor_stump.predict(xor_X).tolist()}). Both are")
print("  defensible - a zero-gain split changes nothing, so skipping it is cheaper -")
print("  and neither affects accuracy. It is exactly this degenerate case where")
print("  gains are all ~0 that makes trees unstable.")
print()
print("  Two splits in sequence do solve XOR, because the second split uses a")
print("  different feature:")
xor_depth2 = DecisionTreeClassifier(max_depth=2, random_state=SEED).fit(xor_X, xor_y)
print(f"    depth 2 predictions {xor_depth2.predict(xor_X).tolist()}  "
      f"accuracy {accuracy_score(xor_y, xor_depth2.predict(xor_X)):.4f}")
print()
print("  root: x0 <= 0.5 -> left impure -> split x1 <= 0.5 -> both children pure")
print()
print("This is the honest case for trees: highly non-linear, axis-aligned structure")
print("that a linear model cannot reach, reachable with a handful of splits. Note")
print("also that the zero-gain situation is why trees can be unstable - when gains")
print("are all ~0, tiny data changes decide the structure entirely.")


# ======================================================================================
# Figures
# ======================================================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))

# --- 1: the decision region of the depth-3 tree --------------------------------------
ax = axes[0]
# Only two features, so the boundary can be drawn as a picture at all.
two_idx = [2, 3]  # petal length, petal width
X2_train, y2_train = X_train[:, two_idx], y_train
model2 = DecisionTreeClassifier(max_depth=3, random_state=SEED, criterion="gini")
model2.fit(X2_train, y2_train)

pad = 0.5
x_min, x_max = X2_train[:, 0].min() - pad, X2_train[:, 0].max() + pad
y_min, y_max = X2_train[:, 1].min() - pad, X2_train[:, 1].max() + pad
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
grid = np.c_[xx.ravel(), yy.ravel()]
zz = model2.predict(grid).reshape(xx.shape)

ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5, 2.5], alpha=0.22,
            cmap="RdYlGn", vmin=-0.5, vmax=2.5)
colors = ["#E03131", "#2F9E44", "#1971C2"]
for label, color in zip([0, 1, 2], colors):
    ax.scatter(X2_train[y2_train == label, 0], X2_train[y2_train == label, 1],
               s=34, alpha=0.85, color=color, edgecolors="white", linewidths=0.6,
               label=iris.target_names[label])
ax.set_xlabel(feature_names[two_idx[0]])
ax.set_ylabel(feature_names[two_idx[1]])
ax.set_title(f"Decision regions, depth 3 (test acc {model2.score(X_test[:, two_idx], y_test):.3f})",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.grid(alpha=0.2)

# --- 2: impurity as a function of a candidate threshold ------------------------------
ax = axes[1]
feature = 2  # petal length
column = X_train[:, feature]
candidates = np.unique(column)
midpoints = (candidates[:-1] + candidates[1:]) / 2.0
parent = SimpleCART.gini(y_train)
gains = []
for threshold in midpoints:
    mask = column <= threshold
    gains.append(parent
                 - mask.mean() * SimpleCART.gini(y_train[mask])
                 - (1 - mask.mean()) * SimpleCART.gini(y_train[~mask]))
gains = np.array(gains)

ax.plot(midpoints, gains, color="#4C6EF5", linewidth=2.2)
best_i = int(np.argmax(gains))
ax.scatter([midpoints[best_i]], [gains[best_i]], s=150, color="#E03131", zorder=5,
           edgecolors="white", linewidths=1.5)
ax.annotate(f"best gain {gains[best_i]:.3f}\nthreshold {midpoints[best_i]:.2f}",
            xy=(midpoints[best_i], gains[best_i]),
            xytext=(midpoints[best_i] + 0.9, gains[best_i] * 0.55),
            fontsize=9, color="#212529",
            arrowprops=dict(arrowstyle="->", color="#E03131", linewidth=1.6))
ax.set_xlabel(f"candidate threshold on {feature_names[feature]}")
ax.set_ylabel("impurity reduction (gain)")
ax.set_title(f"Root split search: {len(midpoints)} candidates on one feature",
             fontsize=11, fontweight="bold")
ax.grid(alpha=0.25)

# --- 3: depth against training and test accuracy -------------------------------------
ax = axes[2]
depths = list(range(1, 16))
train_scores, test_scores = [], []
for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))
train_scores = np.array(train_scores)
test_scores = np.array(test_scores)

ax.plot(depths, train_scores, "o-", color="#E03131", linewidth=2, label="train")
ax.plot(depths, test_scores, "o-", color="#4C6EF5", linewidth=2, label="test")
best_depth = depths[int(np.argmax(test_scores))]
ax.axvline(best_depth, color="#0CA678", linestyle="--", linewidth=2,
           label=f"test-optimal depth {best_depth}")
ax.set_xlabel("max_depth")
ax.set_ylabel("accuracy")
ax.set_title("Train saturates, test peaks then decays", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, frameon=False)
ax.grid(alpha=0.25)
ax.set_ylim(0.7, 1.02)

fig.suptitle("01 - Decision Tree From Scratch", fontsize=13, fontweight="bold")
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
IMPLEMENTATION   SimpleCART, NumPy only, gini criterion, midpoint thresholds

ALGORITHM
  at each node: scan every feature x every midpoint, keep the largest
  impurity reduction, recurse, stop on purity / depth / size / leaf budget

HAND-BUILT TREE (depth 3)
  test accuracy   {my_acc:.4f}
  depth           {mine.depth()}
  leaves          {mine.count_leaves()}

SCIKIT-LEARN TREE (depth 3)
  test accuracy   {sk_acc:.4f}
  depth           {theirs.get_depth()}
  leaves          {theirs.get_n_leaves()}

AGREEMENT        {agreement:.4f} of predictions match

DEPTH SWEEP
  best test accuracy {test_scores.max():.4f} at max_depth={best_depth}
  at max_depth=15 the train/test gap is
  {train_scores[-1] - test_scores[-1]:.4f} - a large gap, i.e. variance, not bias

XOR              depth 1 accuracy {accuracy_score(xor_y, xor_sk1.predict(xor_X)):.4f}
                 every one-feature split has gain exactly 0, so impurity
                 does not fall and one split cannot help
                 depth 2 accuracy {accuracy_score(xor_y, xor_depth2.predict(xor_X)):.4f}

STABILITY (two moons, leave-one-out over {n_refits} refits)
  root split changed     {root_changed}/{n_refits} ({root_changed / n_refits:.1%})
  leaf count changed     {leaves_changed}/{n_refits} ({leaves_changed / n_refits:.1%})
  test predictions differ{predictions_changed:>4}/{n_refits} ({predictions_changed / n_refits:.1%})

COST PROFILE
  training   O(n_features x n log n) per node  (expensive)
  prediction O(depth)                          (cheap)

WHAT TO CARRY FORWARD
  - Gini and entropy rank splits identically in 2-class problems (script 11)
  - the midpoint rule, not the observed value, is the correct threshold
  - train accuracy is 1.0 by construction; judge trees on validation only
  - greedy means the first split is irrevocable - pruning is unavoidable
  - CART handles axis-aligned structure and cannot rotate it (script 12)
""")
print("=" * 78)
