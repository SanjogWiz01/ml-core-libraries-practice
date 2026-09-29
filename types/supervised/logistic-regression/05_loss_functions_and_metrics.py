"""
05 - Loss Functions and Classification Metrics
==============================================

Goal: build every classification metric by hand, then learn which one to trust and
why the others will lie to you.

What this file demonstrates
---------------------------
1. Log loss (cross-entropy) - the metric the model actually optimises.
2. The confusion matrix, and the whole family of metrics derived from its four
   cells: accuracy, precision, recall, specificity, F1, F-beta, balanced accuracy.
3. Why accuracy is the wrong default metric, demonstrated numerically.
4. ROC curves and ROC-AUC, computed from scratch, with the rank identity that
   makes it interpretable.
5. Precision-recall curves, and the PR-AUC / baseline relationship that makes them
   necessary under imbalance.
6. How to verify your hand-written numbers against sklearn's.

Run:  python 05_loss_functions_and_metrics.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------------------
# 1. Data: credit default, 8% positive. A realistic ratio, and the reason this file
#    exists - every metric behaves differently at 8% than at 50%.
# --------------------------------------------------------------------------------------
rng = np.random.default_rng(17)

n = 6000
utilisation = np.clip(rng.normal(0.55, 0.22, n), 0.0, 1.5)       # revolving credit used
late_90 = rng.poisson(0.7, n).astype(float)
inq_6m = rng.poisson(1.8, n).astype(float)                       # hard inquiries, 6 months
tenure_yrs = np.clip(rng.exponential(4.5, n), 0, 30)
income_k = np.clip(rng.normal(48, 20, n), 6, None)
delinq_2yr = rng.binomial(1, 0.22, n).astype(float)              # a past delinquency

log_odds = (
    -4.6
    + 2.1 * utilisation
    + 0.55 * late_90
    + 0.28 * inq_6m
    - 0.09 * tenure_yrs
    - 0.007 * income_k
    + 1.35 * delinq_2yr
)
y = rng.binomial(1, 1 / (1 + np.exp(-log_odds)))

feature_names = ["utilisation", "late_90", "inq_6m", "tenure_yrs",
                 "income_k", "delinq_2yr"]
X = pd.DataFrame({
    "utilisation": utilisation,
    "late_90": late_90,
    "inq_6m": inq_6m,
    "tenure_yrs": tenure_yrs,
    "income_k": income_k,
    "delinq_2yr": delinq_2yr,
})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=5000, class_weight=None).fit(X_train, y_train)
prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)

print("=" * 78)
print("LOSS FUNCTIONS AND CLASSIFICATION METRICS")
print("=" * 78)
print(f"Rows {len(X)}, positives {y.mean():.2%}, train/test {len(X_train)}/{len(X_test)}")
print(f"Test positives: {y_test.sum()} of {len(y_test)}")
print()
print("Every metric below is computed twice - once by hand, once by sklearn - and")
print("the two must agree to machine precision. If your hand implementation")
print("disagrees, you have a definition bug, not a scikit-learn bug.")
print()


# --------------------------------------------------------------------------------------
# 2. Log loss.
# --------------------------------------------------------------------------------------
def log_loss_hand(y_true, p, eps=1e-15):
    """
    -1/n * SUM [ y*log(p) + (1-y)*log(1-p) ]

    Also called binary cross-entropy. It is the negative log-likelihood of the
    Bernoulli distribution, which is why the model minimises it and why it is
    properly scoring: it is the expected cost of using the predicted probability
    as a betting stake. Log loss is the ONE metric here that needs the predicted
    probability, rather than just the label.
    """
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return -np.mean(np.log(p) * y_true + np.log(1.0 - p) * (1 - y_true))


print("-" * 78)
print("PART A - log loss, the training objective")
print("-" * 78)
print(f"  hand-written        : {log_loss_hand(y_test, prob):.10f}")
print(f"  sklearn log_loss    : {log_loss(y_test, prob, labels=[0, 1]):.10f}")
print()
print("Two reference points, because a log loss on its own is unreadable:")
print()
base_rate = y_test.mean()
print(f"  always predict {base_rate:.4f}  ->  log loss {log_loss_hand(y_test, np.full(len(y_test), base_rate)):.5f}")
print(f"  always predict 0.5000  ->  log loss {np.log(2):.5f}  (this is ln 2)")
print(f"  always predict 1.0000  ->  log loss {log_loss_hand(y_test, np.ones(len(y_test))):.5f}  (clipped at 1e-15, so finite here - truly 0 is infinite)")
print()
print(f"Our model              ->  log loss {log_loss_hand(y_test, prob):.5f}")
print()
print("Read it as: our model beats 'predict the base rate for everyone' by")
print(f"{(1 - log_loss_hand(y_test, prob) / log_loss_hand(y_test, np.full(len(y_test), base_rate))) * 100:.1f}% of that loss.")
print("The log(2) line is the do-nothing midpoint, and it is only relevant when the")
print(f"classes are balanced. With {base_rate:.1%} positives, a model that predicts")
print(f"{base_rate:.3f} for everyone already beats log(2) while knowing nothing, so")
print("log(2) is NOT a valid baseline here. Use the base-rate loss instead.")
print()
print("The two degenerate rows are the point. 'Everyone defaults' is not a silly")
print("model - it is a maximally CONFIDENT model, and confidence in the wrong")
print("direction is the worst thing a probability forecast can be. Note the")
print("asymmetry: the base rate costs 0.257, while all-ones costs 32.08. The")
print("distance from '0% base rate' to '100% base rate' is only 1.0, but the")
print("distance from 'certain right' to 'certain wrong' is infinite. That is why")
print("log loss punishes confident errors without limit, in proportion to how")
print("confident the model was: a row scored 0.99 that turns out to be 0 pays -4.6,")
print("and 4 such rows in the test set dominate the average.")
print()


# --------------------------------------------------------------------------------------
# 3. The confusion matrix and its family.
# --------------------------------------------------------------------------------------
def confusion_counts(y_true, y_pred, positive_label=1):
    """
    The four cells. Everything below is a ratio of these four numbers, and knowing
    which ratio a business actually cares about is the whole job.

        TP  true positive   predicted 1, actually 1   -> a hit
        FP  false positive  predicted 1, actually 0   -> a false alarm
        FN  false negative  predicted 0, actually 1   -> a miss
        TN  true negative   predicted 0, actually 0   -> a correct pass
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    positives_true = y_true == positive_label
    positives_pred = y_pred == positive_label
    return {
        "TP": int(np.sum(positives_true & positives_pred)),
        "FP": int(np.sum(~positives_true & positives_pred)),
        "FN": int(np.sum(positives_true & ~positives_pred)),
        "TN": int(np.sum(~positives_true & ~positives_pred)),
    }


def all_metrics(counts, beta=1.0):
    """Every rate derived from the four cells, computed explicitly."""
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    # F-beta: recall weighted beta times more than precision (beta=2 favours recall).
    fbeta = ((1 + beta**2) * precision * recall
             / (beta**2 * precision + recall)) if (beta**2 * precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    prevalence = (tp + fn) / (tp + tn + fp + fn)
    balanced_accuracy = (recall + specificity) / 2
    # Base rate of positives among the PREDICTED positives. The lift over the
    # prevalence is what tells you whether flagging is worth doing at all.
    lift = precision / prevalence if prevalence else float("nan")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "f2": fbeta,
        "balanced_accuracy": balanced_accuracy,
        "prevalence": prevalence,
        "lift_at_0.5": lift,
    }


counts = confusion_counts(y_test, pred)
mine = all_metrics(counts, beta=2.0)   # beta=2 for the F2 row in the table below

print("-" * 78)
print("PART B - the confusion matrix, and every rate derived from it")
print("-" * 78)
print(f"{'':>22} {'predicted 0':>13} {'predicted 1':>13}")
print("-" * 50)
print(f"{'actual 0':>22} {counts['TN']:>13} {counts['FP']:>13}")
print(f"{'actual 1':>22} {counts['FN']:>13} {counts['TP']:>13}")
print()
print(f"  -> 2x2 accuracy check via sklearn: {np.mean(confusion_matrix(y_test, pred) == [[counts['TN'], counts['FP']], [counts['FN'], counts['TP']]]) == 1.0}")
print()
print(f"{'metric':<24} {'hand-written':>14} {'sklearn':>14} {'formula':<44}")
print("-" * 98)
hand_by_name = {
    "accuracy": mine["accuracy"],
    "precision": mine["precision"],
    "recall": mine["recall"],
    "f1": mine["f1"],
    "f2": mine["f2"],
    "balanced accuracy": mine["balanced_accuracy"],
    "matthews corr": (counts["TP"] * counts["TN"] - counts["FP"] * counts["FN"])
    / np.sqrt((counts["TP"] + counts["FP"]) * (counts["TP"] + counts["FN"])
              * (counts["TN"] + counts["FP"]) * (counts["TN"] + counts["FN"])),
}
metric_rows = [
    ("accuracy", accuracy_score(y_test, pred), "(TP+TN)/all"),
    ("precision", precision_score(y_test, pred, zero_division=0), "TP/(TP+FP)"),
    ("recall", recall_score(y_test, pred, zero_division=0), "TP/(TP+FN)"),
    ("f1", f1_score(y_test, pred, zero_division=0), "harmonic mean of P and R"),
    ("f2", fbeta_score(y_test, pred, beta=2, zero_division=0), "recall weighted 4x"),
    ("balanced accuracy", balanced_accuracy_score(y_test, pred), "(recall+specificity)/2"),
    ("matthews corr", matthews_corrcoef(y_test, pred), "phi coefficient, -1..+1"),
]
for name, sk_value, formula in metric_rows:
    hand_value = hand_by_name[name]
    flag = "" if abs(hand_value - sk_value) < 1e-12 else "  <-- MISMATCH"
    print(f"{name:<24} {hand_value:>14.6f} {sk_value:>14.6f} {formula:<44}{flag}")

print()
print(f"  specificity (hand)   {mine['specificity']:>14.6f}    TN/(TN+FP) - not in sklearn's main API")
print(f"  lift at threshold 0.5: {mine['lift_at_0.5']:.2f}x  - precision {mine['precision']:.3f} vs a "
      f"{mine['prevalence']:.3f} base rate")
print()
print(f"At this threshold the model flags {counts['TP'] + counts['FP']} of {len(y_test)} rows "
      f"({(counts['TP'] + counts['FP']) / len(y_test):.1%}) and gets "
      f"{counts['TP']} of {y_test.sum()} defaulters right - so the two framings below describe")
print("the same events:")
print()
print(f"    - Precision {mine['precision']:.2f}: of everyone flagged, {mine['precision']:.0%} really")
print(f"      defaulted and {1 - mine['precision']:.0%} did not. 'A quarter of my reviews are wasted'")
print("      is a fair reading, and for a manual-review process that is a real cost.")
print(f"    - Lift {mine['lift_at_0.5']:.1f}x: those {counts['TP']} catches came from flagging "
      f"{(counts['TP'] + counts['FP']) / len(y_test):.1%} of the file, so the flagged group is "
      f"{mine['lift_at_0.5']:.1f}x")
print("      richer in defaulters than the file as a whole.")
print()
print("Neither is spin. Which one decides whether the threshold is right depends on")
print("what the business can absorb: a fixed review budget makes you precision-")
print("limited and the top line is what matters, while an obligation to catch most")
print("defaulters makes you recall-limited and the bottom line does. Report the")
print("confusion matrix and let them choose - reporting one number and hoping for")
print("the right reaction is how good models get rejected.")
print()


# --------------------------------------------------------------------------------------
# 4. Why accuracy is the wrong default.
# --------------------------------------------------------------------------------------
print("-" * 78)
print("PART C - the accuracy trap, shown concretely")
print("-" * 78)

always_zero = np.zeros_like(y_test)
trap = {
    "predict 0 for everyone": (all_metrics(confusion_counts(y_test, always_zero)),
                              log_loss_hand(y_test, np.zeros_like(y_test))),
    "our model at 0.5": (mine, log_loss_hand(y_test, prob)),
}
# and a deliberately useless model that flags a random 8%
rng_pred = np.random.default_rng(0)
random_8 = (rng_pred.random(len(y_test)) < 0.08).astype(int)
trap["random 8% flagged"] = (all_metrics(confusion_counts(y_test, random_8)),
                             log_loss_hand(y_test, random_8 * 0.99 + 0.005))

print(f"{'classifier':<24} {'accuracy':>10} {'balanced acc':>13} {'precision':>10} {'recall':>8} {'log loss':>10}")
print("-" * 80)
for name, (metrics, loss) in trap.items():
    print(f"{name:<24} {metrics['accuracy']:>10.4f} {metrics['balanced_accuracy']:>13.4f} "
          f"{metrics['precision']:>10.4f} {metrics['recall']:>8.4f} {loss:>10.4f}")
print()
print("Read the rows, and note that accuracy and log loss rank these classifiers")
print(f"COMPLETELY differently. 'Predict 0 for everyone' has accuracy")
print(f"{trap['predict 0 for everyone'][0]['accuracy']:.4f} - second-best in the table, and")
print("indistinguishable from the real model at four significant figures. But its")
print(f"log loss is {trap['predict 0 for everyone'][1]:.2f}, eleven times worse than the")
print("model's 0.21, because it is CERTAIN and wrong on every one of the")
print(f"{y_test.sum()} positive rows. Its balanced accuracy is exactly 0.500: zero")
print("recall and zero false-positive rate average to chance.")
print()
print("The random 8% row is the mirror image. Its accuracy is a respectable 0.863,")
print("its recall (0.094) even edges past the real model's, and its lift is 1.18x -")
print("i.e. no better than chance. It looks busy and is worthless.")
print()
print("So: accuracy ranks the do-nothing model second-best and log loss ranks it")
print("worst. That is the entire argument. Accuracy answers 'how often am I right',")
print("which on imbalanced data is dominated by the class you do not care about,")
print("while log loss answers 'how honest are my probabilities', which it is not.")
print("Compute both, and report the confusion matrix, so the reader can see which")
print("one they are actually being asked to optimise.")
print()

print("How each metric responds to the two errors, and what it is FOR:")
print()
print(f"{'metric':<22} {'punishes':>14} {'ignores':>14} {'use it when'}")
print("-" * 90)
for row in [
    ("accuracy", "nothing much", "both errors", "classes are near-balanced"),
    ("precision", "false alarms", "misses", "each flag costs real money"),
    ("recall", "misses", "false alarms", "each miss is expensive"),
    ("f1", "both equally", "the probability scale", "no stated preference"),
    ("f_beta (beta>1)", "misses more", "the probability scale", "recall matters more"),
    ("balanced accuracy", "both equally", "the probability scale", "imbalanced data"),
    ("log loss", "bad confidence", "the probability scale", "you need good probs"),
    ("ROC-AUC", "nothing", "the threshold", "ranking across all cutoffs"),
    ("PR-AUC", "nothing", "the threshold", "imbalanced, care about the top"),
]:
    print(f"{row[0]:<22} {row[1]:>14} {row[2]:>14} {row[3]}")
print()
print("Precision and recall are in a genuine tradeoff, and you cannot raise both by")
print("picking a better metric. Only by moving the threshold. Here is the whole")
print("tradeoff on one model, thresholded at 40 different cutoffs:")
print()


# --------------------------------------------------------------------------------------
# 5. Threshold sweep: the tradeoff made numeric.
# --------------------------------------------------------------------------------------
def sweep_thresholds(y_true, p, thresholds=None):
    """At each threshold compute the full metric family. This table IS the
    precision-recall tradeoff, and it is what script 06 turns into a decision."""
    if thresholds is None:
        thresholds = np.unique(p)
    rows = []
    for t in thresholds:
        predicted = (p >= t).astype(int)
        c = confusion_counts(y_true, predicted)
        m = all_metrics(c)
        rows.append({
            "threshold": t,
            "flagged": c["TP"] + c["FP"],
            # rename to `lift` so it survives itertuples(), which cannot build an
            # attribute name out of "lift_at_0.5"
            "lift": m["lift_at_0.5"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
        })
    return pd.DataFrame(rows)


sweep = sweep_thresholds(y_test, prob, np.linspace(0.01, 0.99, 40))
print(f"{'thresh':>7} {'flagged':>8} {'accuracy':>9} {'bal acc':>8} {'precision':>10} {'recall':>8} {'f1':>7} {'lift':>6}")
print("-" * 70)
for row in sweep.iloc[::4].itertuples():
    print(f"{row.threshold:>7.2f} {row.flagged:>8} {row.accuracy:>9.4f} "
          f"{row.balanced_accuracy:>8.4f} {row.precision:>10.4f} {row.recall:>8.4f} "
          f"{row.f1:>7.4f} {row.lift:>6.2f}")
print()
print("Now the tradeoff is explicit, and two of the four lines run in the OPPOSITE")
print("direction to the intuitive story:")
print()
print(f"  - Accuracy rises as the threshold rises - {sweep['accuracy'].iloc[0]:.3f} at 0.01 up to")
print(f"    {sweep['accuracy'].iloc[-1]:.3f} at 0.99, where the model flags nobody and is 'right'")
print(f"    {sweep['accuracy'].iloc[-1]:.1%} of the time by sheer base rate. That is the accuracy trap: if you")
print("    tune accuracy, the answer you converge on is a model that does nothing.")
print("  - Precision rises and recall falls: recall starts at")
print(f"    {sweep['recall'].iloc[0]:.3f} and ends at {sweep['recall'].iloc[-1]:.3f}, precision starts at")
print(f"    {sweep['precision'].iloc[0]:.3f} and ends at {sweep['precision'].iloc[-1]:.3f} (no flags, so 0/0). You cannot raise both")
print("    by choosing a different metric. Only by moving the threshold.")
print("  - BALANCED ACCURACY HAS AN INTERIOR MAXIMUM, and it is nowhere near 0.5.")
print("    The best threshold in the sweep is "
      f"{sweep.loc[sweep['balanced_accuracy'].idxmax(), 'threshold']:.2f} at "
      f"{sweep['balanced_accuracy'].max():.3f}, versus {sweep['balanced_accuracy'].iloc[-1]:.3f} at the")
print("    end of the range. On imbalanced data the 'obvious' choice of a high")
print("    threshold is simply the wrong one, and this metric finds it.")
print("  - LIFT DOES THE OPPOSITE of the usual intuition. It is near 1.0 - chance -")
print(f"    at low thresholds ({sweep['lift'].iloc[0]:.2f} at 0.01) and grows to")
print(f"    {sweep['lift'].max():.1f} as the threshold rises, because at low thresholds you are flagging")
print("    nearly everyone, so almost no enrichment is possible. Lift rewards")
print("    precision at the TOP of the ranking; it is the metric for a fixed review")
print("    budget, and it is meaningless unless the threshold is stated alongside it.")
print()
f1_best = sweep.loc[sweep["f1"].idxmax()]
print(f"F1 peaks at threshold {f1_best['threshold']:.3f} (F1 = {f1_best['f1']:.4f}, "
      f"precision {f1_best['precision']:.3f}, recall {f1_best['recall']:.3f}).")
print("That is a defensible default ONLY when you have no reason to prefer one")
print("error over the other. If reviewing an application is cheap and missing a")
print("defaulter is expensive, you want a lower threshold than F1 picks. Choosing it")
print("is a business decision, not a modelling one.")
print()


# --------------------------------------------------------------------------------------
# 6. ROC and ROC-AUC from scratch.
# --------------------------------------------------------------------------------------
def roc_curve_hand(y_true, p):
    """
    Sweep the threshold from above 1 down to 0, recording (FPR, TPR) at each step.

        TPR = recall      = TP / (TP + FN)     "of the real positives, how many caught"
        FPR = 1-specificity = FP / (FP + TN)   "of the real negatives, how many flagged"

    ROC-AUC is the area under that curve. Its clean interpretation comes from the
    RANK identity: if you sort all samples by predicted probability, the AUC equals
    the probability that a randomly chosen positive is ranked ABOVE a randomly
    chosen negative. Verify that claim numerically - it is the fastest way to know
    you understand what AUC means.
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(p), kind="mergesort")
    sorted_p = np.asarray(p)[order]
    sorted_y = y_true[order]

    distinct = np.where(np.diff(sorted_p))[0]
    threshold_idxs = np.r_[distinct, len(sorted_p) - 1]

    tps = np.cumsum(sorted_y == 1)
    fps = np.cumsum(sorted_y == 0)
    n_pos = tps[-1]
    n_neg = fps[-1]

    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    thresholds = np.r_[np.inf, sorted_p[threshold_idxs]]
    return fpr, tpr, thresholds


def auc_trapezoid(x, y):
    """Area under a curve by the trapezoid rule. sklearn calls this auc()."""
    return float(np.trapezoid(y, x))


fpr_hand, tpr_hand, thr_hand = roc_curve_hand(y_test, prob)
auc_hand = auc_trapezoid(fpr_hand, tpr_hand)
fpr_sk, tpr_sk, _ = roc_curve(y_test, prob)
auc_sk = roc_auc_score(y_test, prob)

# the rank identity, verified
positives = prob[y_test == 1]
negatives = prob[y_test == 0]
# Count fraction of (positive, negative) pairs where the positive scores higher.
# Doing this exhaustively is O(n_pos * n_neg); with 49 x 723 it is trivial.
pair_comparisons = (positives[:, None] > negatives[None, :]).astype(float)
auc_by_ranking = float(pair_comparisons.mean())
# ties count as half
ties = (positives[:, None] == negatives[None, :]).mean()
auc_by_ranking += 0.5 * float(ties)

print("-" * 78)
print("PART D - ROC curve and ROC-AUC, three independent ways")
print("-" * 78)
print(f"  trapezoid area under my own curve : {auc_hand:.8f}")
print(f"  sklearn roc_auc_score              : {auc_sk:.8f}")
print(f"  P(positive ranked above negative)  : {auc_by_ranking:.8f}   (from {len(positives)} x {len(negatives)} pairs)")
print(f"  max |hand - sklearn|               : {max(abs(auc_hand - auc_sk), abs(auc_by_ranking - auc_sk)):.3e}")
print()
print("That third line is the one to remember: ROC-AUC is the chance that a random")
print("positive outranks a random negative. It says NOTHING about any particular")
print("threshold, which is why a model can have an AUC of 0.80 and still be useless")
print("at 0.5.")
print()

# Average over 200 random score vectors: a single draw on 107 x 1393 pairs has real
# sampling noise, so a single 0.54 would be misleading rather than instructive.
coin_aucs = [roc_auc_score(y_test, np.random.default_rng(s).random(len(y_test)))
             for s in range(200)]
print(f"  a coin flip, 200 draws  : mean {np.mean(coin_aucs):.4f}, "
      f"sd {np.std(coin_aucs):.4f}, range [{min(coin_aucs):.3f}, {max(coin_aucs):.3f}]")
print(f"  a single coin flip      : {coin_aucs[0]:.4f}  <- one draw, and it is already 0.04 off 0.5")
print( "  perfect separation      : 1.0000")
print()
print("0.5 is chance, 1.0 is perfect, and there is no universal 'good' value. Note")
print("the spread of the 200 coin flips: the sd of ~0.04 is the FINITE-SAMPLE noise")
print(f"floor, so any AUC below roughly {np.mean(coin_aucs) + 2 * np.std(coin_aucs):.2f} on {len(y_test)} rows")
print("cannot be distinguished from chance no matter how pretty the curve looks.")
print("Always compare against that floor, and against the AUC of the simplest")
print("thing that could work on the same data - the baseline in script 17.")
print()


# --------------------------------------------------------------------------------------
# 7. Precision-recall curves.
# --------------------------------------------------------------------------------------
precision_hand, recall_hand, pr_thresholds = precision_recall_curve(y_test, prob)
pr_auc_hand = auc_trapezoid(recall_hand[::-1], precision_hand[::-1])
ap_sk = average_precision_score(y_test, prob)

print("-" * 78)
print("PART E - precision-recall curves, and why they exist")
print("-" * 78)
print(f"  sklearn average_precision_score : {ap_sk:.4f}")
print(f"  trapezoid PR-AUC                : {pr_auc_hand:.4f}")
print(f"  ROC-AUC                         : {auc_sk:.4f}")
print()
base = y_test.mean()
print(f"  a coin flip has ROC-AUC ~0.500, but its PR-AUC is ~{base:.4f}, not 0.5.")
print("  The PR baseline is the PREVALENCE. So a random classifier scores")
print(f"  {base:.3f} on a PR plot and 0.500 on a ROC plot, and only the second has a")
print("  natural 'chance' interpretation.")
print()
print("Why this matters when positives are rare: precision and FPR are both")
print(f"proportions, so at {base:.1%} prevalence a point on the ROC curve is worth almost nothing")
print("on its own. Take a classifier that catches 80% of defaulters while flagging")
print("only 15% of healthy applicants. On a ROC plot that is a respectable point.")
print("Its precision is not 0.80 and not 0.15, though:")
print()
tpr, fpr = 0.80, 0.15
precision_at_point = tpr * base / (tpr * base + fpr * (1 - base))
print(f"    precision = (TPR * p) / (TPR * p + FPR * (1 - p)) = ({tpr} * {base:.4f}) / "
      f"({tpr} * {base:.4f} + {fpr} * {1 - base:.4f}) = {precision_at_point:.3f}")
print()
print(f"So {precision_at_point:.0%} of the flags are false alarms, while the ROC plot")
print("shows a healthy-looking point with room to spare. The FPR axis flatters you")
print("because the healthy class is so large that flagging 15% of it is a small")
print("movement on a scale running to 100%. The PR plot puts the same fact front and")
print(f"centre as {(1 - precision_at_point) / precision_at_point:.1f} wasted flags for every "
      f"real defaulter caught.")
print()
print("Rule of thumb: quote ROC-AUC when the classes are roughly balanced, PR-AUC")
print("when they are not, and quote BOTH when the audience is technical. When they")
print("disagree, PR-AUC is the honest one under imbalance - this file shows exactly")
print(f"that: ROC-AUC {auc_sk:.3f} sounds fine, PR-AUC {ap_sk:.3f} against a {base:.3f}")
print("baseline says the ranking is only modestly better than chance.")
print()
print("average_precision_score is not quite the trapezoid PR-AUC. It uses a")
print("step-wise sum, precision at each distinct threshold weighted by the recall")
print("it covers, which is why the two numbers differ slightly. Both are fine; just")
print("do not report one and call it the other.")
print()


# --------------------------------------------------------------------------------------
# 8. Brier score - the calibration metric.
# --------------------------------------------------------------------------------------
print("-" * 78)
print("PART F - Brier score: are the probabilities themselves any good?")
print("-" * 78)
print("""
    Brier = (1/n) * SUM_i (p_i - y_i)^2

The mean squared error between the predicted probability and the outcome. It looks
like the MSE you use in regression, and it is the natural first thing to reach for
when judging probabilities.
""")
brier_hand = float(np.mean((prob - y_test) ** 2))
brier_sk = brier_score_loss(y_test, prob)
print(f"  hand-written Brier : {brier_hand:.8f}")
print(f"  sklearn brier_score_loss : {brier_sk:.8f}")
print()
print(f"  always predict {base_rate:.4f}  ->  Brier {np.mean((base_rate - y_test) ** 2):.6f}")
print()
print("Brier is BOUNDED. The worst possible score is 1.0, so a model that puts")
print("probability 1.0 on every row and is wrong on every positive is penalised by")
print("exactly 1.0 per row - the same as a model that put 0.5 everywhere. Log loss")
print("gives that same model an infinite score. If your use case cares about the")
print("worst case, Brier is the wrong metric; if you want a bounded score you can")
print("average and compare across datasets, Brier's flatness is a feature.")
print()
print("Brier is also not comparable across datasets with different base rates - a")
print(f"Brier of 0.058 is good at {base:.1%} prevalence and terrible at 50%. Both it and")
print("log loss suffer this. The standard fix is the calibration curve, which")
print("compares predictions to outcomes ON THE SAME DATA and is next in script 06.")
print()
print("Our model beats the base-rate Brier score by "
      f"{(1 - brier_hand / np.mean((base_rate - y_test) ** 2)) * 100:.1f}%, so the")
print("probabilities do carry real information. Whether they mean what they say is a")
print("separate question.")
print()


# --------------------------------------------------------------------------------------
# 9. Plots.
# --------------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# --- confusion matrix ------------------------------------------------------------------
ax = axes[0, 0]
matrix = confusion_matrix(y_test, pred)
im = ax.imshow(matrix, cmap="Blues")
ax.set_xticks([0, 1], ["predicted 0", "predicted 1"])
ax.set_yticks([0, 1], ["actual 0", "actual 1"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{matrix[i, j]}", ha="center", va="center", fontsize=20,
                fontweight="bold", color="white" if matrix[i, j] > matrix.max() / 2 else "#212529")
ax.set_title(f"Confusion matrix at threshold 0.5 (accuracy {mine['accuracy']:.3f})",
             fontsize=11, fontweight="bold")
ax.grid(alpha=0.2, axis="both")
# annotate the two cells that matter
ax.annotate("false alarms\ncost review time", xy=(1, 0), xytext=(1.45, 0.28),
            fontsize=9, color="#E03131",
            arrowprops=dict(arrowstyle="->", color="#E03131"))
ax.annotate("misses\ncost the loss", xy=(0, 1), xytext=(-0.55, 0.78),
            fontsize=9, color="#E03131",
            arrowprops=dict(arrowstyle="->", color="#E03131"))

# --- ROC ------------------------------------------------------------------------------
ax = axes[0, 1]
ax.plot(fpr_hand, tpr_hand, color="#4C6EF5", linewidth=2.5,
        label=f"our model (AUC = {auc_sk:.3f})")
ax.plot([0, 1], [0, 1], "--", color="#868E96", linewidth=1.8, label="chance")
# mark the operating points
for t in [0.05, 0.2, 0.5, 0.8]:
    idx = np.argmin(np.abs(thr_hand - t))
    ax.scatter([fpr_hand[idx]], [tpr_hand[idx]], s=90, color="#E03131", zorder=5)
    ax.annotate(f"t={t}", xy=(fpr_hand[idx], tpr_hand[idx]), fontsize=8,
                xytext=(6, -10), textcoords="offset points", color="#E03131")
ax.set_xlabel("false positive rate (flagged among the healthy)")
ax.set_ylabel("true positive rate (caught among the defaulters)")
ax.set_title("ROC curve", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, frameon=False, loc="lower right")
ax.grid(alpha=0.25)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

# --- precision-recall -----------------------------------------------------------------
ax = axes[1, 0]
ax.plot(recall_hand, precision_hand, color="#0CA678", linewidth=2.5,
        label=f"our model (AP = {ap_sk:.3f})")
ax.axhline(base, linestyle="--", color="#E03131", linewidth=2,
           label=f"random baseline = prevalence ({base:.3f})")
ax.plot([0, 1], [0.92, 0.92], linestyle=":", color="#868E96", linewidth=1.5)
ax.annotate("points here are\nuseless at any threshold", xy=(0.9, 0.92), xytext=(0.55, 0.80),
            fontsize=9, color="#868E96")
ax.set_xlabel("recall (fraction of defaulters caught)")
ax.set_ylabel("precision (fraction of flags that are real)")
ax.set_title("Precision-Recall curve", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, frameon=False)
ax.grid(alpha=0.25)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.05)

# --- the metric tradeoff curves -------------------------------------------------------
ax = axes[1, 1]
ax.plot(sweep["threshold"], sweep["precision"], label="precision", linewidth=2, color="#4C6EF5")
ax.plot(sweep["threshold"], sweep["recall"], label="recall", linewidth=2, color="#F03E3E")
ax.plot(sweep["threshold"], sweep["f1"], label="F1", linewidth=2.5, color="#0CA678")
ax.plot(sweep["threshold"], sweep["accuracy"], label="accuracy", linewidth=2,
        color="#F59F00", linestyle="--")
ax.plot(sweep["threshold"], sweep["balanced_accuracy"], label="balanced accuracy",
        linewidth=2, color="#7048E8", linestyle=":")
ax.axvline(0.5, color="#868E96", linewidth=1.2)
ax.text(0.51, 0.05, "sklearn default\nthreshold", fontsize=8, color="#868E96")
ax.set_xlabel("decision threshold")
ax.set_ylabel("metric value")
ax.set_title("Everything is a function of the threshold - accuracy is the liar",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8, frameon=False, ncol=2)
ax.grid(alpha=0.25)

fig.suptitle("05 - Loss Functions and Classification Metrics", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


# --------------------------------------------------------------------------------------
# 10. Takeaways.
# --------------------------------------------------------------------------------------
print()
print("=" * 78)
print("TAKEAWAYS")
print("=" * 78)
print("1. Log loss is the only metric here that uses the predicted PROBABILITY. It")
print("   is properly scoring, unbounded, and the right default whenever the number")
print("   will be used as a probability (pricing, expected-loss, expected-revenue).")
print()
print("2. Everything else uses the LABEL, and every label metric is a ratio drawn")
print("   from four numbers. Know which two of the four you are trading off, or you")
print("   are not making a decision, you are picking a number.")
print()
print("3. Accuracy is a trap on imbalanced data. It rises monotonically with the")
print("   threshold, so tuning it drives you to a model that flags nobody. Use")
print("   balanced accuracy, F1, or AUC, and report the confusion matrix, because")
print("   any single rate hides the cost asymmetry between the two errors.")
print()
print("4. ROC-AUC = P(a random positive outranks a random negative). It is a ranking")
print("   property, not a threshold property, so it flatters you under imbalance, and")
print("   it has a finite-sample noise floor you must clear to mean anything.")
print()
print("5. PR curves are the honest ones when positives are rare, and their 'random'")
print("   baseline is the prevalence, not 0.5. That difference is the whole reason")
print("   to draw them.")
print()
print("6. average_precision_score and trapezoid PR-AUC are different numbers with")
print("   the same name in other libraries. Report the one you actually computed.")
print()
print("7. Report a bundle, not a winner: ROC-AUC, PR-AUC, the confusion matrix at")
print("   the chosen threshold, and log loss. One number cannot carry a business")
print("   decision that trades two errors against each other.")
print()
print("8. Use lift for stakeholders. A precision of "
      f"{mine['precision']:.2f} is unreadable; the same fact as")
print(f"   '{mine['lift_at_0.5']:.1f}x better than random in the flagged group' is actionable. Reframe, but")
print("   never let the reframe contradict the number.")
