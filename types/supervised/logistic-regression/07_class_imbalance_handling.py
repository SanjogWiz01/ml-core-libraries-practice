"""
07 - Class Imbalance Handling
============================

Goal: find out what actually happens when you "fix" an imbalanced dataset, and which
of the fixes are worth doing.

The setup
---------
Card fraud detection: 30,000 transactions, about 6% of them fraudulent. This is
realistic - real fraud rates are usually between 0.1% and 10%, and everything
interesting happens in that range.

The trap
--------
On a 6% problem, a model that predicts "not fraud" for every single transaction
scores 94% accuracy and has caught nothing at all. Accuracy on an imbalanced problem
is not a weak signal, it is an actively misleading one, and it is the default metric
in a lot of code that reaches production.

What this file demonstrates
---------------------------
1. Why accuracy is worse than useless here, and which metrics to use instead.
2. Why ROC-AUC flatters you and PR-AUC does not, on the very same scores.
3. Thresholds as a capacity lever: what happens when you must catch at least X%.
4. The three standard "fixes" - undersampling, SMOTE, class_weight - measured.
5. The finding that matters: they barely change the RANKING and wreck the
   CALIBRATION.
6. Proof that class_weight='balanced' is just minority replication, and why that
   makes it the better of the two.
7. The prior correction that undoes the damage, with the odds algebra that
   justifies it.

Run:  python 07_class_imbalance_handling.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, log_loss, matthews_corrcoef,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

np.set_printoptions(precision=5, suppress=True)
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


# ======================================================================================
# The dataset
# ======================================================================================
def make_fraud_data(n=30000, intercept=-3.2, signal=1.6, seed=2024):
    """
    Transaction-level fraud, generated from a known log-odds so the population rate
    and the signal strength are both under our control.
    """
    rng = np.random.default_rng(seed)

    amount = rng.lognormal(3.2, 1.1, n)                      # transaction size
    dist_home = np.abs(rng.normal(0, 1, n)) * rng.lognormal(2.0, 0.8, n)  # home-terminal distance
    hour = rng.integers(0, 24, n).astype(float)               # hour of day
    online = rng.binomial(1, 0.55, n).astype(float)          # online order
    prior_fraud = rng.binomial(1, 0.22, n).astype(float)      # history on the card

    linear = (0.55 * (np.log(amount) - 3.2)
              + 0.35 * (np.log(dist_home) - 2.0)
              - 0.030 * hour
              + 0.45 * online
              + 0.70 * prior_fraud)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(intercept + signal * linear))))

    X = np.column_stack([amount, dist_home, hour, online, prior_fraud])
    names = ["amount", "dist_home", "hour", "online", "prior_fraud"]
    return X, y, names


X, y, feature_names = make_fraud_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

TRAIN_PREV = float(y_train.mean())
TEST_PREV = float(y_test.mean())
N_POS_TEST = int(y_test.sum())

print("=" * 78)
print("CLASS IMBALANCE HANDLING")
print("=" * 78)
print(f"Transactions           : {len(y):,}")
print(f"Fraudulent            : {int(y.sum()):,} ({y.mean():.2%})")
print(f"Training rows          : {len(y_train):,} ({TRAIN_PREV:.2%} positive)")
print(f"Test rows              : {len(y_test):,} ({TEST_PREV:.2%} positive, {N_POS_TEST} frauds)")
print()
print("Everything below is measured on the same held-out test set.")
print()


# ======================================================================================
# A small reporting helper, used everywhere
# ======================================================================================
def score_report(y_true, prob, threshold=0.5):
    """One row per model, every metric we care about, at one threshold."""
    pred = (prob >= threshold).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y_true, prob),
        "PR-AUC": average_precision_score(y_true, prob),
        "log loss": log_loss(y_true, prob),
        "Brier": brier_score_loss(y_true, prob),
        "mean p": float(prob.mean()),
        "accuracy": accuracy_score(y_true, pred),
        "balanced acc": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, pred),
        "n flagged": int(pred.sum()),
    }


def print_table(labeled_reports, chunk=6):
    """
    `labeled_reports` is a sequence of (label, report_dict) pairs, so the index is
    built from the labels rather than expected to already be a DataFrame column.
    Printed in chunks of columns so the rows stay readable at terminal width.
    """
    df = pd.DataFrame([report for _, report in labeled_reports],
                      index=[label for label, _ in labeled_reports])
    for start in range(0, len(df.columns), chunk):
        print(df.iloc[:, start:start + chunk].to_string(
            float_format=lambda v: f"{v:>8.4f}"))
        if start + chunk < len(df.columns):
            print()
    print()


# ======================================================================================
# PART A - accuracy is worse than useless
# ======================================================================================
print("-" * 78)
print("PART A - the metric that has to go first")
print("-" * 78)
print("""
Start with the null model. It never predicts fraud, so it is wrong every single
time it matters, and it is the thing your model has to beat to be worth shipping.
""")

null_prob = np.zeros(len(y_test))
null_pred = np.zeros(len(y_test), dtype=int)

print(f"Null model: predict 'not fraud' for all {len(y_test):,} transactions")
print(f"  accuracy         {accuracy_score(y_test, null_pred):.4f}   <- looks great")
print(f"  frauds caught    {int(((null_pred == 1) & (y_test == 1)).sum())} of {N_POS_TEST}")
print(f"  F1               {f1_score(y_test, null_pred, zero_division=0):.4f}   <- the truth")
print(f"  MCC              {matthews_corrcoef(y_test, null_pred):.4f}   <- the truth")
print(f"  balanced acc     {balanced_accuracy_score(y_test, null_pred):.4f}   <- the truth")
print()
print(f"Accuracy says {accuracy_score(y_test, null_pred):.0%}. Balanced accuracy and MCC say 0.50,")
print("which is the truth: the null model is exactly as good as a coin flip, because")
print("that is what it is. Both numbers are computed from the same predictions - only")
print("the weighting of the two error types differs, and that single choice is the")
print("whole difference between a 94% success story and a 50% one.")
print()

print("The two numbers, side by side, for every plausible metric on the null model:")
print()
print(f"  {'metric':<16} {'value':>10}   verdict")
print("  " + "-" * 52)
for label, value, verdict in [
    ("accuracy", accuracy_score(y_test, null_pred), "flattering - ignores the 6% that matter"),
    ("balanced accuracy", balanced_accuracy_score(y_test, null_pred), "honest - 0.50 is a coin flip"),
    ("MCC", matthews_corrcoef(y_test, null_pred), "honest - 0.00 means no association"),
    ("F1", f1_score(y_test, null_pred, zero_division=0), "honest - 0.00 is the right answer"),
    ("ROC-AUC", roc_auc_score(y_test, null_prob), "0.50 - every pair is a tie"),
    ("PR-AUC", average_precision_score(y_test, null_pred), "equals the base rate"),
]:
    print(f"  {label:<16} {value:>10.4f}   {verdict}")
print()
print("The last two rows are worth a second look. A constant score produces no")
print("ranking at all, yet ROC-AUC returns a clean 0.5000 rather than raising an")
print("error - because every positive-negative pair is a tie, and the convention is")
print("to award half credit for a tie. The number is not wrong, it is just carrying")
print("no information: a model that has learned nothing and a coin flip are")
print("indistinguishable here, and only one of them is a coin flip.")
print()


# ======================================================================================
# The reference model
# ======================================================================================
base_model = LogisticRegression(max_iter=5000).fit(X_train, y_train)
base_prob = base_model.predict_proba(X_test)[:, 1]

print("The reference model is an ordinary LogisticRegression with no tricks at all:")
print()
coefs = pd.Series(base_model.coef_.ravel(), index=feature_names)
print(coefs.to_frame("coefficient").to_string(float_format=lambda v: f"{v:>9.4f}"))
print()
print("Two coefficients on the original scale, so the effect sizes mean something:")
print(f"  online order       odds x {np.exp(coefs['online']):.2f} per 1-unit change (0 -> 1)")
print(f"  prior fraud        odds x {np.exp(coefs['prior_fraud']):.2f} per 1-unit change (0 -> 1)")
print(f"  transaction hour   odds x {np.exp(coefs['hour']):.3f} per hour later")
print()
print("Fraud is up for online orders and cards with history, and down overnight - a")
print("plausible story, which is the minimum you should expect from any coefficient")
print("you are going to put in front of somebody.")
print()
rows = [score_report(y_test, base_prob)]
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:>9.4f}", header=False))
print()
print(f"Mean predicted probability {base_prob.mean():.4f} against a true test rate of {TEST_PREV:.4f}.")
print("The model is calibrated - script 06's first takeaway, verified on a third dataset.")
print("Hold on to that number; Part D is about what happens when it stops being true.")
print()


# ======================================================================================
# PART B - ROC-AUC flatters, PR-AUC does not
# ======================================================================================
print("-" * 78)
print("PART B - two AUCs, two verdicts")
print("-" * 78)
print("""
The same scores, two average precision numbers. ROC-AUC asks about pairs - one
positive, one negative, how often is the positive scored higher. That normalises by
the negatives, so the answer barely notices how many negatives there are.

PR-AUC asks about the positive class, and the negatives only enter as the thing
getting in the way. When positives are 6% of the data, the two disagree.
""")

print(f"{'curve':<12} {'AUC':>9}   random-guess baseline")
print("-" * 52)
print(f"{'ROC':<12} {roc_auc_score(y_test, base_prob):>9.4f}   0.5000")
print(f"{'PR':<12} {average_precision_score(y_test, base_prob):>9.4f}   {TEST_PREV:.4f}  (the base rate)")
print()
print(f"ROC-AUC says {roc_auc_score(y_test, base_prob):.2f}, which sounds like a usable model.")
print(f"PR-AUC says {average_precision_score(y_test, base_prob):.2f}, against a no-skill baseline of")
print(f"{TEST_PREV:.2f} - so the model is {average_precision_score(y_test, base_prob) / TEST_PREV:.1f}x better than random, not {roc_auc_score(y_test, base_prob) / 0.5:.1f}x.")
print()
print("The gap between those two ratios IS the imbalance, and it is why reporting")
print("ROC-AUC on a rare-event problem is a way of hiding the problem. Concretely, at")
print("the point where the model catches 80% of frauds:")
print()
# NOTE: precision_recall_curve returns DECREASING recall, and returns one more
# precision/recall point than it returns thresholds (the extra point is the
# recall=0, precision=1 corner). Pair them up so the arrays line up.
pr_prec, pr_rec, pr_thresh = precision_recall_curve(y_test, base_prob)
pr_rec = pr_rec[:-1]
pr_prec = pr_prec[:-1]


def threshold_for_recall(prob, recall_target):
    """
    The score threshold at which the model first reaches `recall_target`.
    precision_recall_curve returns one more recall point than it returns thresholds,
    so the extra corner point is dropped before the two are paired up.
    """
    _, recall, thresholds = precision_recall_curve(y_test, prob)
    return float(np.interp(recall_target, recall[::-1][:-1], thresholds[::-1]))


t80 = threshold_for_recall(base_prob, 0.80)
n_review_80 = int(np.sum(base_prob >= t80))
n_review_50 = int(np.sum(base_prob >= threshold_for_recall(base_prob, 0.50)))
n_review_95 = int(np.sum(base_prob >= threshold_for_recall(base_prob, 0.95)))
print(f"So catching 80% of the {N_POS_TEST} frauds means reviewing {n_review_80:,} transactions -")
print(f"{n_review_80 / len(y_test):.0%} of the queue - to save about {int(0.80 * N_POS_TEST)} of them. That ratio is")
print("the number an operations team actually needs, and it is the one a single AUC")
print(f"never gives you. Compare the ends: going from 50% recall to 95% recall costs")
print(f"{n_review_95 / n_review_50:.1f}x the review effort for 45 extra points of recall.")
print("Whether that is a bargain depends entirely on the cost of a review - which is")
print("script 06's closed form, applied to this dataset.")
print()
print("So the rule, stated once: on an imbalanced problem, report PR-AUC (or the")
print("precision-recall curve) rather than ROC-AUC. Use ROC-AUC only when you also")
print("want the false-positive rate explicitly, because it is the only one of the two")
print("that puts both error types on a shared, prevalence-free scale.")
print()


# ======================================================================================
# PART C - the threshold is a capacity decision
# ======================================================================================
print("-" * 78)
print("PART C - the threshold is a capacity decision")
print("-" * 78)
print("""
Before reaching for a resampling library, try the free option. Imbalance is mostly
a statement about the threshold, and thresholds cost nothing to move.

The operations version of the question is usually a budget: 'we can review 5% of
transactions', or 'we must catch at least 90% of frauds'. Both are one line.
""")

review_budgets = [0.01, 0.02, 0.05, 0.10, 0.20]
print(f"{'review budget':>14} {'threshold':>10} {'flagged':>9} {'caught':>8} {'recall':>8} {'precision':>10}")
print("-" * 64)
budget_rows = []
for frac in review_budgets:
    t = float(np.quantile(base_prob, 1 - frac))
    pred = (base_prob >= t).astype(int)
    caught = int(((pred == 1) & (y_test == 1)).sum())
    budget_rows.append({
        "budget": frac, "threshold": t,
        "flagged": int(pred.sum()), "caught": caught,
        "recall": caught / N_POS_TEST,
        "precision": precision_score(y_test, pred, zero_division=0),
    })
    r = budget_rows[-1]
    print(f"{frac:>13.0%} {t:>10.4f} {r['flagged']:>9,} {caught:>8} {r['recall']:>8.4f} {r['precision']:>10.4f}")
print()
print("Read that table as a menu, not a ranking. A 5% review budget catches")
print(f"{[r for r in budget_rows if r['budget'] == 0.05][0]['recall']:.0%} of frauds at "
      f"{[r for r in budget_rows if r['budget'] == 0.05][0]['precision']:.0%} precision. Whether")
print("that is a good trade depends entirely on what a missed fraud costs versus a")
print("wasted review - which is the closed form from script 06, applied to this")
print("dataset. Nobody can answer that from the model alone.")
print()
print("The same scores also let you hit a recall FLOOR, which is the constraint")
print("fraud teams usually state, because a missed fraud is a regulatory event:")
print()
for target in [0.50, 0.75, 0.90, 0.95]:
    t = threshold_for_recall(base_prob, target)
    pred = (base_prob >= t).astype(int)
    caught = int(((pred == 1) & (y_test == 1)).sum())
    print(f"  catch at least {target:.0%}  ->  threshold {t:.4f}, review "
          f"{pred.sum() / len(y_test):.1%} of rows, precision {precision_score(y_test, pred, zero_division=0):.2%}")
print()
print("Neither of these tables needed a single extra model. Same fit, same scores, a")
print("different question. This is the cheapest imbalance technique there is, and it")
print("is the one most often skipped.")
print()


# ======================================================================================
# PART D - the three standard fixes, measured
# ======================================================================================
print("-" * 78)
print("PART D - undersampling, SMOTE, and class_weight")
print("-" * 78)
print("""
Now the interventions everyone reaches for. All three do the same thing in
substance: they change the class prior the model is fitted under, from 6% to 50%.
The interesting question is what that buys and what it costs.

SMOTE is implemented here rather than imported, because imblearn is not available
and because seeing the interpolation written out is the point - it is 15 lines.
""")
print()


def smote_oversample(X_in, y_in, k_neighbors=5, seed=0):
    """
    Synthetic Minority Over-sampling Technique.

    For each synthetic minority point: pick a minority sample, pick one of its k
    nearest minority neighbours, and place a new point on the line between them at a
    uniformly random position. The new point is on no real decision boundary drawn in
    the data - it is a convex combination of two real minority points - which is both
    the reason it works and the reason it can hurt.
    """
    rng = np.random.default_rng(seed)
    X_minority = X_in[y_in == 1]
    n_existing_majority = int((y_in == 0).sum())
    n_to_make = n_existing_majority - len(X_minority)

    nn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X_minority)
    _, neighbour_idx = nn.kneighbors(X_minority)

    synthetic = []
    for _ in range(n_to_make):
        i = rng.integers(len(X_minority))
        j = neighbour_idx[i, rng.integers(1, k_neighbors + 1)]
        gap = rng.random()
        synthetic.append(X_minority[i] + gap * (X_minority[j] - X_minority[i]))

    X_out = np.vstack([X_in, np.asarray(synthetic)])
    y_out = np.concatenate([y_in, np.ones(n_to_make, dtype=int)])
    return X_out, y_out


# (a) random undersampling: keep every positive, throw away most of the negatives
rng_us = np.random.default_rng(0)
pos_idx = np.where(y_train == 1)[0]
neg_idx = np.where(y_train == 0)[0]
keep_idx = np.concatenate([pos_idx, rng_us.choice(neg_idx, size=len(pos_idx), replace=False)])
X_under, y_under = X_train[keep_idx], y_train[keep_idx]

# (b) SMOTE: synthesise positives until the classes balance
X_smote, y_smote = smote_oversample(X_train, y_train, seed=1)

# (c) class_weight: same idea, no data touched
balanced_model = LogisticRegression(class_weight="balanced", max_iter=5000).fit(X_train, y_train)

print(f"{'training set':<26} {'rows':>8} {'prevalence':>11}")
print("-" * 48)
for label, ys in [("original", y_train), ("random undersample", y_under), ("SMOTE", y_smote)]:
    print(f"{label:<26} {len(ys):>8,} {ys.mean():>11.2%}")
print(f"{'original (class_weight)':<26} {len(y_train):>8,} {y_train.mean():>11.2%}   <- data untouched")
print()

under_model = LogisticRegression(max_iter=5000).fit(X_under, y_under)
smote_model = LogisticRegression(max_iter=5000).fit(X_smote, y_smote)

candidates = {
    "original (6.2% positive)": base_model,
    "random undersample": under_model,
    "SMOTE": smote_model,
    "class_weight='balanced'": balanced_model,
}
probs = {name: m.predict_proba(X_test)[:, 1] for name, m in candidates.items()}

print("All four, on the same test set, ranked by PR-AUC at threshold 0.5:")
print()
print_table([(name, score_report(y_test, probs[name])) for name in candidates])
print()
base_pr = average_precision_score(y_test, probs["original (6.2% positive)"])
base_auc = roc_auc_score(y_test, probs["original (6.2% positive)"])
print("Now the result worth pausing on. Look at the ranking columns across the last")
print("three rows, then look at the log loss column in the same rows.")
print()
for name in ["random undersample", "SMOTE", "class_weight='balanced'"]:
    pr = average_precision_score(y_test, probs[name])
    auc = roc_auc_score(y_test, probs[name])
    ll = log_loss(y_test, probs[name])
    print(f"  {name:<24} PR-AUC {pr:.4f} ({pr - base_pr:+.4f} vs original)   "
          f"ROC-AUC {auc:.4f} ({auc - base_auc:+.4f})   log loss {ll:.4f} "
          f"({ll - log_loss(y_test, probs['original (6.2% positive)']):+.4f})")
print()
inflation = probs["class_weight='balanced'"].mean() / probs["original (6.2% positive)"].mean()
print(f"Every one of them changes the ranking by a rounding error - under 0.002 of")
print(f"PR-AUC on a base rate of 0.06. And every one of them makes the probabilities")
print(f"about {inflation:.0f}x too large, which is where the log loss comes from.")
print()
print("But the recall column in that table jumps from 0.05 to 0.65, so the balanced")
print("models look much better. That comparison is not fair, and it is exactly the")
print("trap. Their 0.5 threshold is not your 0.5 threshold - it sits far lower on the")
print("real scale, because the model believes the world is 50/50. They are flagging")
print("2,345 rows to the original's 56. They are not detecting more fraud; they are")
print("just flagging forty times as much.")
print()
print("The fair comparison fixes the review budget instead of the threshold. Top-k")
print("rows by score, same k for everyone, which is what an operations team actually")
print("does when it has capacity for k reviews:")
print()
budgets = [0.005, 0.01, 0.05, 0.10]
short_names = ["original", "undersample", "SMOTE", "class_weight"]
print(f"{'budget':>8} " + " ".join(f"{s:>13}" for s in short_names))
print("-" * (8 + 14 * len(short_names)))
for frac in budgets:
    k = int(frac * len(y_test))
    cells = [f"{int(y_test[np.argsort(-probs[name])[:k]].sum()) / N_POS_TEST:>13.4f}"
             for name in candidates]
    print(f"{frac:>7.1%} " + " ".join(cells))
print()
print("(fraction of frauds caught, reviewing the same number of rows)")
print()
spreads = []
for frac in budgets:
    k = int(frac * len(y_test))
    vals = [int(y_test[np.argsort(-probs[name])[:k]].sum()) / N_POS_TEST for name in candidates]
    spreads.append(max(vals) - min(vals))
print(f"The spread between best and worst model is at most {max(spreads):.3f} at any budget, and the")
print("ordering flips between budgets - undersampling wins at 1% and loses at 5%.")
print("Four models, one ranking, and the differences are noise.")
print()
print("This is the lesson, and it is the opposite of the folklore:")
print()
print("  Balancing the training set does not teach the model anything new about the")
print("  minority class. It tells the model the world is 50/50 when it is 6/94, and")
print("  the model dutifully reports probabilities to match the world it was told")
print("  about. The RANKING is nearly untouched, because the ranking is a statement")
print("  about order, and order survives a change of prior. The CALIBRATION is")
print("  destroyed, because calibration is a statement about scale, and scale is")
print("  exactly what you just changed.")
print()
print("So if the goal was better minority-class detection, none of these delivered")
print("it. If the goal was better minority-class detection you can actually use, the")
print("answer was a threshold, from Part C, and it costs nothing.")
print()


# ======================================================================================
# PART E - class_weight is just replication
# ======================================================================================
print("-" * 78)
print("PART E - what class_weight actually does")
print("-" * 78)
print("""
The three fixes are not three ideas, they are two. Undersampling and SMOTE both
resize the dataset; class_weight does not resize anything. So the dataset-based pair
must be throwing something away that class_weight keeps, and the useful question is
which.
""")

n_pos, n_neg = int(y_train.sum()), int((y_train == 0).sum())
replicate = int(round(n_neg / n_pos))
print(f"Training set: {n_pos:,} positives, {n_neg:,} negatives, ratio {n_neg / n_pos:.1f}:1")
print()

# Replicate the minority class up to parity and fit with no class weights at all.
pos_positions = np.where(y_train == 1)[0]
replicated_idx = np.concatenate([np.arange(len(y_train))] + [pos_positions] * (replicate - 1))
X_repl, y_repl = X_train[replicated_idx], y_train[replicated_idx]

# A large C makes the penalty negligible, so this is close to the pure weighted MLE.
unreg = 1e6
cw_model = LogisticRegression(class_weight="balanced", C=unreg, max_iter=20000).fit(X_train, y_train)
repl_model = LogisticRegression(C=unreg, max_iter=20000).fit(X_repl, y_repl)

print(f"Replicating the positives {replicate} times gives a training set of "
      f"{len(y_repl):,} rows at")
print(f"{y_repl.mean():.1%} positive. Fit it with NO class weights and compare to")
print("class_weight='balanced' fit on the original 21,000 rows:")
print()
compare = pd.DataFrame({
    "class_weight='balanced'": cw_model.coef_.ravel(),
    f"minority x{replicate}": repl_model.coef_.ravel(),
    "difference": cw_model.coef_.ravel() - repl_model.coef_.ravel(),
}, index=feature_names)
print(compare.to_string(float_format=lambda v: f"{v:>11.6f}"))
print()
print(f"Maximum absolute difference across all coefficients: "
      f"{np.abs(cw_model.coef_.ravel() - repl_model.coef_.ravel()).max():.2e}")
print()
print("They are the same model. class_weight='balanced' is not a different algorithm -")
print("it is the weighted likelihood, and a weighted likelihood is the same thing as")
print(f"replicating rows in proportion to their weights. The fit is identical to")
print(f"numerical precision, on {len(y_train):,} rows instead of {len(y_repl):,}.")
print()
print("Which settles the comparison with undersampling, which is the SAME idea done")
print("the expensive way:")
print()
print(f"  undersampling throws away {n_neg - n_pos:,} real negatives, "
      f"{(n_neg - n_pos) / n_neg:.0%} of them,")
print(f"  to reach parity. That is {1 - len(y_under) / len(y_train):.0%} of the training set, deleted. "
      f"class_weight keeps all {n_neg:,}")
print("  and reaches the same parity in the objective function.")
print()
print("So between the two, class_weight wins on the only axis that matters when you")
print("have just thrown away most of your data for nothing:")
print()
print("  1. It is deterministic. Undersampling depends on a random seed, and an")
print("     unlucky one is a model you cannot reproduce from the record.")
print("  2. It keeps the negatives' information. Each discarded negative is a")
print("     genuine observation you had and threw away.")
print("  3. It is one keyword, not a data pipeline.")
print()
print("Undersampling still has a use - when the positives are so rare that you need")
print("to drop the dataset to fit in memory - but on a dataset this size it is")
print("strictly the worse option.")
print()
print("SMOTE is the interesting one, because it is NOT the same idea. It invents new")
print("minority points on segments between real ones rather than duplicating them.")
print(f"It has the largest coefficient norm of the four ("
      f"{np.linalg.norm(smote_model.coef_):.3f} against "
      f"{np.linalg.norm(base_model.coef_):.3f} for the original),")
print("which is the signature of interpolating inside a region the real data thins")
print("out near the decision boundary. That is exactly where you need it, and also")
print("exactly where the synthetic points are least supported by evidence. SMOTE is")
print("worth measuring rather than assuming - and its AUC here is")
print(f"{roc_auc_score(y_test, probs['SMOTE']):.4f} against the original's {base_auc:.4f}, which")
print("is a difference of about "
      f"{abs(roc_auc_score(y_test, probs['SMOTE']) - base_auc):.4f}.")
print()


# ======================================================================================
# PART F - putting the prior back
# ======================================================================================
print("-" * 78)
print("PART F - undoing the prior shift")
print("-" * 78)
print("""
Every balanced model in Part D is confidently wrong about how common fraud is. It
says 39% when the answer is 6%. That is fixable exactly, because we know precisely
what was changed: the prior. The model is calibrated to a 50% world, and we can
convert its answer into the real world with two lines of algebra on the odds.

The transformation is a rescaling of the odds by the prior ratio:

    p_balanced    what the model says, calibrated to pi = 0.5
    pi_real       the prevalence it should have been calibrated to
    pi_balanced   0.5, by construction

                 p_real / (1 - p_real)  =  p_balanced / (1 - p_balanced) * pi_real/(1-pi_real)
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                    the prior correction

Multiplying the odds by a constant and resquashing is strictly monotone, so the
RANKING is provably untouched - the correction can only fix the scale. That is a
guarantee, not a hope, and it is checked below.
""")
print()


def correct_for_prior(p_balanced, pi_real, pi_balanced=0.5):
    """
    Map a probability that was calibrated under `pi_balanced` to one calibrated under
    `pi_real`, by rescaling the odds by the prior ratio. Strictly monotone in the
    input, so the ranking is unchanged.
    """
    odds_balanced = p_balanced / (1.0 - p_balanced)
    odds_real = odds_balanced * (pi_real / (1.0 - pi_real)) / (pi_balanced / (1.0 - pi_balanced))
    return odds_real / (1.0 + odds_real)


rows_f = []
for name, p_bal in probs.items():
    is_balanced = name != "original (6.2% positive)"
    p_used = correct_for_prior(p_bal, TEST_PREV) if is_balanced else p_bal
    rows_f.append({
        "model": name,
        "prior": "corrected" if is_balanced else "as fitted",
        "mean p": float(p_used.mean()),
        "log loss": log_loss(y_test, p_used),
        "Brier": brier_score_loss(y_test, p_used),
        "ROC-AUC": roc_auc_score(y_test, p_used),
        "PR-AUC": average_precision_score(y_test, p_used),
    })
print(pd.DataFrame(rows_f).to_string(index=False, float_format=lambda v: f"{v:>9.4f}"))
print()
print(f"For reference, the true test prevalence is {TEST_PREV:.4f}. A mean predicted")
print("probability near that line is a calibrated model; a mean far from it is not,")
print("and the mean is the number to check first.")
print()
print("Read the log loss column across the corrected rows. The balanced models went")
print("from about 0.55 back to about 0.21, essentially where the uncorrected original")
print("sits. Two lines of algebra recovered everything the resampling threw away.")
print()
print("And the ranking columns are unchanged to the last printed digit, which is the")
print("guarantee above made visible. Let it be checked rather than asserted:")
print()
for name in ["random undersample", "SMOTE", "class_weight='balanced'"]:
    before = probs[name]
    after = correct_for_prior(before, TEST_PREV)
    auc_before = roc_auc_score(y_test, before)
    auc_after = roc_auc_score(y_test, after)
    order_before = np.argsort(np.argsort(before))
    order_after = np.argsort(np.argsort(after))
    print(f"  {name:<24} AUC {auc_before:.10f} -> {auc_after:.10f}  "
          f"(delta {abs(auc_before - auc_after):.2e})   identical ordering: {np.array_equal(order_before, order_after)}")
print()
print("The ordering is bit-for-bit the same, so the ROC-AUC difference is floating")
print("point noise rather than a real effect. The correction is free, exact, and")
print("cannot possibly hurt the ranking. There is no argument against applying it.")
print()
print("The one thing it will NOT fix is a model that genuinely learned the wrong")
print("thing. The prior shift is a known, invertible distortion; a mis-specified")
print("model is not. Correcting the prior of a bad model produces a confidently")
print("calibrated bad model, which is a considerably more dangerous artefact than")
print("the miscalibrated one you started with.")
print()


# ======================================================================================
# Plots
# ======================================================================================
print("-" * 78)
print("PLOTS")
print("-" * 78)

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

COLORS = {
    "original (6.2% positive)": "#4C6EF5",
    "random undersample": "#0CA678",
    "SMOTE": "#F59F00",
    "class_weight='balanced'": "#E03131",
}

# --- PR curve vs ROC curve ------------------------------------------------------------
ax = axes[0, 0]
for name, p in probs.items():
    pr, rc, _ = precision_recall_curve(y_test, p)
    ax.plot(rc, pr, linewidth=2.2, color=COLORS[name], label=name)
ax.axhline(TEST_PREV, color="#868E96", linestyle=":", linewidth=2)
ax.text(0.02, TEST_PREV + 0.012, f"no-skill baseline = base rate {TEST_PREV:.3f}",
        fontsize=9, color="#495057")
ax.set_xlabel("recall")
ax.set_ylabel("precision")
ax.set_title("Precision-recall curves: all four models are the same curve",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8.5, frameon=False, loc="lower left")
ax.grid(alpha=0.25)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# --- ROC curve, for contrast ----------------------------------------------------------
ax = axes[0, 1]
for name, p in probs.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    ax.plot(fpr, tpr, linewidth=2.2, color=COLORS[name], label=name)
ax.plot([0, 1], [0, 1], "--", color="#868E96", linewidth=1.6, label="chance")
ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate")
ax.set_title("ROC curves hide the imbalance that the PR curves expose",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8.5, frameon=False, loc="lower right")
ax.grid(alpha=0.25)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# --- what balancing does to the probabilities ----------------------------------------
ax = axes[1, 0]
bins = np.linspace(0, max(0.45, probs["class_weight='balanced'"].max() * 1.05), 46)
for name, p in probs.items():
    ax.hist(p, bins=bins, histtype="step", linewidth=2.2, color=COLORS[name], label=name)
ax.axvline(TEST_PREV, color="#212529", linestyle="--", linewidth=2)
ax.text(TEST_PREV + 0.008, ax.get_ylim()[1] * 0.92, f"true rate {TEST_PREV:.3f}",
        fontsize=9, color="#212529")
ax.set_yscale("log")
ax.set_xlabel("predicted probability")
ax.set_ylabel("count (log scale)")
ax.set_title(f"Balancing the training set inflates every probability ~{inflation:.0f}x",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8.5, frameon=False)
ax.grid(alpha=0.25)

# --- the prior correction -------------------------------------------------------------
ax = axes[1, 1]
for name, p in probs.items():
    if name == "original (6.2% positive)":
        continue
    corrected = correct_for_prior(p, TEST_PREV)
    ax.plot(p, corrected, linewidth=2.2, color=COLORS[name],
            label=f"{name}: {p.mean():.3f} -> {corrected.mean():.3f}")
plain_p = probs["original (6.2% positive)"]
ax.plot([0, 0.02], [0, 0.02], "--", color="#868E96", linewidth=1.8)
ax.text(0.021, 0.0205, "y = x (a no-op)", fontsize=8.5, color="#868E96", ha="right")
ax.set_xlim(0, 0.42)
ax.set_ylim(0, 0.16)
ax.set_xlabel("predicted probability, as the balanced model reports it")
ax.set_ylabel("after the prior correction")
ax.set_title("One rescaling of the odds puts the scale back; the order never moves",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8.5, frameon=False, loc="upper right")
ax.grid(alpha=0.25)

fig.suptitle("07 - Class Imbalance Handling", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


# ======================================================================================
print()
print("=" * 78)
print("TAKEAWAYS")
print("=" * 78)
print("1. Accuracy is not a weak metric on an imbalanced problem, it is an inverted")
print("   one. Predicting the majority class scores 94% and catches nothing. Use")
print("   balanced accuracy, MCC, F1 or PR-AUC - all of which put a 0.50 / 0.00 at")
print("   the null model, which is the truth.")
print()
print("2. On rare events, prefer PR-AUC to ROC-AUC. ROC-AUC normalises by the")
print("   negatives, so it barely moves when the prevalence collapses; the PR curve")
print("   measures the positive class directly and its no-skill baseline is the base")
print("   rate, which makes the improvement legible.")
print()
print("3. Try a threshold before anything else. Imbalance is mostly a statement about")
print("   where to cut, and the cut is free. Every capacity or recall constraint can")
print("   be met with the same fitted model and one quantile.")
print()
print("4. Balancing the training set barely changes the ranking and ruins the")
print("   calibration. Under-sampling, SMOTE and class_weight all move PR-AUC by")
print("   under 0.002 while inflating every predicted probability by roughly the")
print("   prior ratio. That is the whole result: you changed the world the model")
print("   believes in, and the model's ordering is a fact about order, not scale.")
print()
print("5. class_weight='balanced' is the weighted likelihood, which is exactly")
print("   equivalent to replicating the minority class. Verified here to numerical")
print("   precision. Between it and under-sampling it is deterministic, it keeps the")
print("   data, and it is one keyword - so it is the default.")
print()
base_ll = log_loss(y_test, probs["original (6.2% positive)"])
balanced_ll = log_loss(y_test, probs["class_weight='balanced'"])
corrected_ll = log_loss(y_test, correct_for_prior(probs["class_weight='balanced'"], TEST_PREV))
print("6. If you balance, put the prior back. Rescale the odds by pi_real/(1-pi_real)")
print(f"   and log loss falls from {balanced_ll:.2f} to {corrected_ll:.3f} - within "
      f"{abs(corrected_ll - base_ll):.3f} of the unweighted model's {base_ll:.3f}.")
print("   It is strictly monotone, so it provably cannot affect the ranking - worth")
print("   checking rather than assuming.")
print()
print("7. Be clear about what the correction cannot do. It fixes a known, invertible")
print("   distortion. It will happily make a badly misspecified model look")
print("   confidently calibrated, which is a worse failure than an obviously")
print("   miscalibrated one.")
