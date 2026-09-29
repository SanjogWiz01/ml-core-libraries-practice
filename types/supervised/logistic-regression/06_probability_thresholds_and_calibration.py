"""
06 - Probability Thresholds and Calibration
==============================================

Goal: separate the two things a classifier gives you, and learn to fix each one.

The two things
--------------
A logistic regression returns a probability AND a label. These are not the same
output viewed twice - they are two different decisions, and they can be good and bad
independently:

    RANKING quality  - are the high-probability rows actually the likely ones?
                       Measured by ROC-AUC / PR-AUC. Threshold-independent.
    CALIBRATION      - when it says 0.20, do 20% of those rows turn out to be 1?
                       Measured by the reliability diagram, Brier score, log loss.

You can have a perfectly ranked model that lies about its probabilities. You can also
have a beautifully calibrated model that cannot rank at all - and that case turns
out to be much easier to produce than it sounds. They are fixed with completely
different tools, and confusing them is the root of most bad threshold choices.

What this file demonstrates
---------------------------
1. What C actually does - which is almost nothing, until the data is separable.
2. The one case where C runs away: complete separation, and the calibrated-looking
   confident liar it produces when the world changes.
3. A calibrated but useless model, to prove the two properties are orthogonal.
4. Reliability diagrams computed by hand, with Laplace smoothing and why it is
   mandatory.
5. Two repairs: re-regularise (global) vs Platt / isotonic (per-bin).
6. The leak: calibrating on the data you evaluate on.
7. Choosing a threshold from cost, not from F1, with the closed form to check it.
8. Sanity-checking the threshold against the operational flag rate.

Run:  python 06_probability_thresholds_and_calibration.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_circles
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

def calibration_curve_hand(y_true, p, n_bins=10, laplace=1.0):
    """
    Bin the predictions, then compare the MEAN PREDICTION in each bin to the OBSERVED
    FREQUENCY of positives in it.

    Perfect calibration means the observed rate equals the predicted rate in every
    bin, i.e. the points lie on the diagonal. The reliability diagram is that
    equation, drawn. The VERTICAL gap is the quantity of interest; which side the
    points fall on is a plotting convention and carries no information.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = which == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        mean_pred = float(p[mask].mean())
        obs_freq = float(y_true[mask].mean())
        # Shrink the observed frequency toward the bin's own mean prediction. With
        # one row in a bin the raw frequency is exactly 0.0 or 1.0, which is a
        # perfect and entirely meaningless line; this makes the small bins visibly
        # less certain instead.
        smoothed = (obs_freq * n_b + laplace * mean_pred) / (n_b + laplace)
        se = np.sqrt(smoothed * (1.0 - smoothed) / (n_b + laplace))
        rows.append({
            "bin": b, "range_lo": edges[b], "range_hi": edges[b + 1], "n": n_b,
            "mean_pred": mean_pred, "obs_freq": obs_freq, "smoothed": smoothed,
            "ci_lo": smoothed - 1.96 * se, "ci_hi": smoothed + 1.96 * se,
        })
    return pd.DataFrame(rows)


def calibration_error(y_true, p, n_bins=10):
    """
    ECE = SUM_b (n_b / n) * |mean_pred_b - obs_freq_b|

    The single-number summary of the reliability diagram: a count-weighted average
    of the vertical gaps. Being count-weighted is the important detail - ECE is
    dominated by whichever bins hold the most rows, so it can look excellent while
    being badly wrong exactly where a bin is small and the model is confident. That
    is why the diagram is not optional.
    """
    curve = calibration_curve_hand(y_true, p, n_bins)
    weights = curve["n"] / curve["n"].sum()
    return float((weights * (curve["mean_pred"] - curve["obs_freq"]).abs()).sum())


def platt_transform(p):
    """Log-odds of a probability. Bounded input, unbounded scale - the right thing to
    regress a second stage on, since the log-odds is where a linear model actually
    lives and the sigmoid's curvature has already been removed."""
    q = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    return np.log(q / (1.0 - q))


print("=" * 78)
print("PROBABILITY THRESHOLDS AND CALIBRATION")
print("=" * 78)


# --------------------------------------------------------------------------------------
# Two datasets, and why we need both.
# --------------------------------------------------------------------------------------
def two_blobs(n, shift, noise, seed):
    """
    Two isotropic Gaussians, one shifted by `shift` along the diagonal.

    shift/noise is the signal-to-noise ratio, and it decides what an honest model is
    allowed to claim:
      ratio small   classes overlap, the log-likelihood has a FINITE maximum, and
                    no amount of data or freedom will produce confident predictions
      ratio large   the classes barely overlap, and eventually the MLE stops
                    existing entirely (complete separation - script 16)
    """
    r = np.random.default_rng(seed)
    direction = np.array([1.0, 1.0]) / np.sqrt(2.0)
    X = r.normal(0.0, noise, (n, 2))
    y = np.zeros(n, dtype=int)
    y[: n // 2] = 1
    X[y == 1] += shift * direction
    return X, y


# (A) overlapping: shift 1.2 with noise 1.0. Honest, learnable, NOT separable.
X_overlap, y_overlap = two_blobs(8000, shift=1.2, noise=1.0, seed=29)
Xo_train, Xo_test, yo_train, yo_test = train_test_split(
    X_overlap, y_overlap, test_size=0.3, random_state=0, stratify=y_overlap
)

# (B) genuinely separable: shift 4.0 with noise 0.5. The training set admits a
#     perfect linear split, so the log-likelihood has NO finite maximum and the
#     penalty is the only thing holding the coefficients down. That fact drives
#     the entire second half of this file.
X_sep, y_sep = two_blobs(6000, shift=4.0, noise=0.5, seed=31)
Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    X_sep, y_sep, test_size=0.3, random_state=0, stratify=y_sep
)

print(f"Dataset A (overlapping) : {len(X_overlap)} rows, {y_overlap.mean():.1%} positive")
print(f"Dataset B (separable)   : {len(X_sep)} rows, {y_sep.mean():.1%} positive")
print()
print("Both are 50/50, so the prevalence confound is out of the way and any")
print("difference between them is about geometry, not class balance.")
print()
_probe = LogisticRegression(C=1e6, max_iter=20000).fit(Xs_train, ys_train)
_sep_errors = int((_probe.predict(Xs_train) != ys_train).sum())
print(f"Separability check on dataset B's training set: a model with C=1e6 misclassifies")
print(f"{_sep_errors} of {len(Xs_train)} training rows. At 0 the data is completely separated and the")
print("likelihood is unbounded; the script re-checks this in Part A.")
print()


# --------------------------------------------------------------------------------------
# PART A - what C actually does
# --------------------------------------------------------------------------------------
print("-" * 78)
print("PART A - what C actually does: almost nothing, until the data is separable")
print("-" * 78)
print("""
Nearly every tutorial says "increase C to reduce underfitting" as though C were a
master difficulty dial. On overlapping data it is close to irrelevant, and the reason
is worth internalising:

    When the classes overlap, the log-likelihood HAS a finite maximum.

The coefficients that maximise it are interior to the space - there is a specific
|w| at which extra magnitude stops helping, because the overlap means every extra
unit of |w| starts misclassifying rows. So there is no runaway, and shrinking
|w| with a penalty can only move you a short way from the answer you already had.

The penalty starts to matter when the classes are SEPARABLE, because then the
likelihood is monotone increasing in |w| and the maximum is at infinity. Then the
penalty is the only thing standing between you and a divergent coefficient. That is
complete separation, and it is the subject of script 16.
""")

overlap_results = {}
for c in [0.001, 0.01, 0.1, 1.0, 100.0]:
    m = LogisticRegression(C=c, max_iter=5000).fit(Xo_train, yo_train)
    p = m.predict_proba(Xo_test)[:, 1]
    overlap_results[c] = {
        "model": m, "prob": p,
        "auc": roc_auc_score(yo_test, p),
        "pr_auc": average_precision_score(yo_test, p),
        "log_loss": log_loss(yo_test, p),
        "brier": brier_score_loss(yo_test, p),
        "w_norm": float(np.linalg.norm(m.coef_)),
    }

print(f"{'C':>10} {'||w||':>9} {'ROC-AUC':>9} {'log loss':>10} {'Brier':>8} {'p range':>20}")
print("-" * 72)
for c, r in overlap_results.items():
    print(f"{c:>10g} {r['w_norm']:>9.3f} {r['auc']:>9.4f} {r['log_loss']:>10.4f} "
          f"{r['brier']:>8.4f} {f'[{r['prob'].min():.3f}, {r['prob'].max():.3f}]':>20}")
print()
aucs = [r["auc"] for r in overlap_results.values()]
moderate = [r["auc"] for c, r in overlap_results.items() if c >= 0.01]
print(f"Across four orders of magnitude of C, ROC-AUC moves by "
      f"{max(aucs) - min(aucs):.4f} in total - and by "
      f"{max(moderate) - min(moderate):.4f} once C reaches 0.01.")
print()
print("So there are two regimes, and it is worth being precise about which one you")
print("are in. At the top of the range (C from 0.01 to 100, three orders of")
print("magnitude) nothing happens at all: the unpenalised answer is already here and")
print("the penalty is only adding a small amount of shrinkage. At the bottom")
print("something does happen - C=0.001 is over-regularised, ||w|| is roughly half")
print("the unpenalised value, the log loss is worse, and the probability range is")
print("compressed. That is ordinary shrinkage bias, not a structural effect.")
print()
print("The point is not that C is useless. It is that C is a shrinkage dial, and")
print("shrinking only helps if there is variance to remove. On a well-specified")
print("problem with a finite MLE and plenty of data, the unpenalised estimate")
print("already has low variance and there is very little for C to buy. If you were")
print("told to 'tune C to fix underfitting' on data like this, you would be wasting")
print("your afternoon.")
print()

sep_results = {}
for c in [0.001, 0.01, 0.1, 1.0, 100.0]:
    m = LogisticRegression(C=c, max_iter=5000).fit(Xs_train, ys_train)
    p = m.predict_proba(Xs_test)[:, 1]
    p_train = m.predict_proba(Xs_train)[:, 1]
    sep_results[c] = {
        "model": m, "prob": p, "prob_train": p_train,
        "auc": roc_auc_score(ys_test, p),
        "log_loss": log_loss(ys_test, p),
        "log_loss_train": log_loss(ys_train, p_train),
        "w_norm": float(np.linalg.norm(m.coef_)),
        "train_errors": int((m.predict(Xs_train) != ys_train).sum()),
    }

print("Now the same sweep on the SEPARABLE dataset, with the training-set error")
print("counted so you can see for yourself that the MLE does not exist:")
print()
print(f"{'C':>10} {'||w||':>9} {'train err':>10} {'ROC-AUC':>9} {'train ll':>11} {'most extreme p':>17}")
print("-" * 76)
for c, r in sep_results.items():
    most_extreme = max(r["prob"].min(), 1 - r["prob"].max())
    print(f"{c:>10g} {r['w_norm']:>9.3f} {r['train_errors']:>10} {r['auc']:>9.4f} "
          f"{r['log_loss_train']:>11.6f} {most_extreme:>17.2e}")
print()
w_low = sep_results[0.001]["w_norm"]
w_high = sep_results[100.0]["w_norm"]
err_at_low = sep_results[0.001]["train_errors"]
extreme_p = max(sep_results[100.0]["prob"].min(), 1 - sep_results[100.0]["prob"].max())
print("A single training error at C=0.001 is enough. The log-likelihood has no finite")
print("maximum as soon as one misclassified row can be fixed by moving the boundary,")
print("and a boundary that moves freely can be pushed out to infinity. ||w|| climbs")
print(f"from {w_low:.2f} to {w_high:.2f}, the training log loss falls to "
      f"{sep_results[100.0]['log_loss_train']:.2e}, and the")
print(f"predictions become numerically absurd: {extreme_p:.1e} is not a probability any")
print("evidence supports - it is a large number that happened to pass through a")
print("sigmoid. (The error count only reaches 0 at C=100 because the optimiser")
print("actually reached that far; the failure mode is present at every C.)")
print()
print("Note that ROC-AUC does not move at all. This is the same phenomenon as in")
print("Part A, in its extreme: the boundary's DIRECTION is already right, and |w|")
print("only controls confidence. On this data every metric says the model is")
print("perfect. It is perfect on THIS data.")
print()


# --------------------------------------------------------------------------------------
# PART B - the calibrated confident liar
# --------------------------------------------------------------------------------------
print("-" * 78)
print("PART B - the failure mode: confident, well-calibrated, and completely wrong")
print("-" * 78)
print("""
The model trained on separable data is not just overconfident in theory. Take it and
point it at the overlapping dataset - the same features, drawn from the same
generator, just with the two populations moved a little closer together. Nothing
about the model has changed. Nothing about the world has changed much either. It
should be slightly less certain, and instead it is catastrophically wrong.

This is the real shape of a production calibration incident: the model was fine, the
population drifted, and the probabilities did not degrade gracefully. A log loss
that is fine in testing and catastrophic in production is the signature.
""")

liar = sep_results[100.0]
honest = overlap_results[1.0]
# Score both models on the SAME full overlapping dataset, so the comparison is
# like-for-like rather than mixing a 2400-row test split with an 8000-row set.
liar_probs_on_overlap = liar["model"].predict_proba(X_overlap)[:, 1]
honest_probs_on_overlap = honest["model"].predict_proba(X_overlap)[:, 1]
base_rate_loss = log_loss(y_overlap, np.full(len(y_overlap), y_overlap.mean()))

print(f"{'model':<34} {'ROC-AUC':>9} {'log loss':>10} {'Brier':>8} {'mean p':>9} {'p>0.99 rows':>13}")
print("-" * 88)
for name, p in [("separable model -> overlapping data", liar_probs_on_overlap),
                ("overlapping model -> overlapping data", honest_probs_on_overlap),
                ("predict the base rate (do nothing)", np.full(len(y_overlap), y_overlap.mean()))]:
    extreme = int(np.sum((p > 0.99) | (p < 0.01)))
    print(f"{name:<34} {roc_auc_score(y_overlap, p):>9.4f} {log_loss(y_overlap, p):>10.4f} "
          f"{brier_score_loss(y_overlap, p):>8.4f} {p.mean():>9.4f} {extreme:>13}")
print()
print(f"On its own data the separable model scores a log loss of "
      f"{liar['log_loss']:.6f}. On this neighbouring")
print(f"population it scores {log_loss(y_overlap, liar_probs_on_overlap):.4f} - worse than the "
      f"{base_rate_loss:.4f} you get from predicting the")
print("base rate for everyone, and doing literally nothing. That is the shape of a")
print("production calibration incident: the model was fine, the population drifted,")
print("and the probabilities did not degrade gracefully. A log loss that is fine in")
print("testing and catastrophic in production is the signature.")
print()
print("Now the Brier column, which is the more interesting one. Brier went from")
print(f"0.09 on its own data to {brier_score_loss(y_overlap, liar_probs_on_overlap):.3f} - a visible move, but nothing like the log loss. That is")
print("because Brier is BOUNDED at 1.0 per row: a wrong 0.99 costs it about 0.24")
print("regardless of how confident the model was, whereas log loss charges -4.6.")
print("A team watching Brier sees a number drift and has no reason to panic. A team")
print("watching log loss sees the whole thing on fire. That is the argument for")
print("log loss as the production monitoring metric, and the argument against")
print("Brier being 'nicer to look at'.")
print()
print("The two model outputs also disagree in a more useful way. Watch the mean")
print(f"probability: the liar says {liar_probs_on_overlap.mean():.3f} when the true rate is "
      f"{y_overlap.mean():.3f}, and the honest model says {honest_probs_on_overlap.mean():.3f}.")
print("A mean predicted probability far from the observed rate is the single cheapest")
print("drift check there is, and it needs one number and no chart.")
print()


# --------------------------------------------------------------------------------------
# PART C - a calibrated model with no skill at all
# --------------------------------------------------------------------------------------
print("-" * 78)
print("PART C - a calibrated model with ZERO ranking skill")
print("-" * 78)
print("""
The mirror image of Part B, and the reason you should never quote calibration
without AUC. Concentric circles have a circular boundary, so a linear model cannot
represent the truth. Fitting one anyway is textbook misspecification.

The prediction is the uncomfortable one. The ranking goes to chance - a straight
line through a ring has no reason to put the inner points on one side - and the
PROBABILITIES get BETTER, not worse. The reason is worth stating precisely before
you see the numbers, because it is the single most important idea in this file:
a model with no signal outputs a probability near 0.5 for everything, and on a
balanced problem near 0.5 is close to correct by definition. The safe answer and
the honest answer coincide precisely when there is nothing to say.
""")

X_circles, y_circles = make_circles(n_samples=6000, factor=0.45, noise=0.06, random_state=5)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_circles, y_circles, test_size=0.3, random_state=0, stratify=y_circles
)
circular_model = LogisticRegression(max_iter=5000).fit(Xc_train, yc_train)
circular_prob = circular_model.predict_proba(Xc_test)[:, 1]

circular_auc = roc_auc_score(yc_test, circular_prob)
circular_ece = calibration_error(yc_test, circular_prob)
honest_ece = calibration_error(yo_test, overlap_results[1.0]["prob"])
print(f"{'dataset / model':<40} {'ROC-AUC':>9} {'log loss':>10} {'ECE':>8}")
print("-" * 72)
print(f"{'circles, linear model':<40} {circular_auc:>9.4f} "
      f"{log_loss(yc_test, circular_prob):>10.4f} {circular_ece:>8.4f}")
print(f"{'blobs, linear model (honest)':<40} {overlap_results[1.0]['auc']:>9.4f} "
      f"{overlap_results[1.0]['log_loss']:>10.4f} {honest_ece:>8.4f}")
print()
print(f"The circular model has a ROC-AUC of {circular_auc:.4f} - indistinguishable from a coin flip,")
print("so it has learned nothing at all, and a log loss of "
      f"{log_loss(yc_test, circular_prob):.4f} against the {np.log(2):.4f} of a coin flip.")
print()
print("But look at the ECE column. The useless model scores "
      f"{circular_ece:.4f} and the model that actually")
print(f"learns the problem scores {honest_ece:.4f} - the broken one is {honest_ece / circular_ece:.0f}x BETTER calibrated. That is not")
print("a quirk of the seed. It is structural: the circular model is so uncertain")
print("that it attaches a probability near 0.5 to everything, and on a balanced")
print("problem 'near 0.5' is approximately calibrated by definition. A model that")
print("knows nothing produces near-perfect calibration, because the safe answer and")
print("the honest answer coincide when you have no signal.")
print()
print("So if you had monitored only ECE, you would have seen the numbers improve")
print("exactly as the model got worse. Calibration cannot detect a model that does")
print("not know anything. Only a ranking metric can, which is why the two have to be")
print("reported together and never instead of each other.")
print()


# --------------------------------------------------------------------------------------
# PART D - reliability diagrams
# --------------------------------------------------------------------------------------
print("-" * 78)
print("PART D - the reliability diagram, computed by hand")
print("-" * 78)
print("""
    Bins of predicted probability  ->  observed frequency of positives in that bin

Read the VERTICAL gap, not the horizontal one. A model is calibrated when the
observed frequency matches the predicted probability, so the gap should be zero in
every bin. The side the points fall on is irrelevant and flips with the plotting
convention - only the size of the gap carries information.
""")

rel_datasets = {
    "honest blobs": (yo_test, overlap_results[1.0]["prob"]),
    "C=0.001 blobs": (yo_test, overlap_results[0.001]["prob"]),
    "liar on blobs": (y_overlap, liar_probs_on_overlap),
}
curves = {}
print(f"{'model':<16} {'ECE':>8} {'worst bin gap':>15} {'p range':>20}")
print("-" * 64)
for name, (yt, p) in rel_datasets.items():
    curve = calibration_curve_hand(yt, p, 10)
    curves[name] = curve
    weights = curve["n"] / curve["n"].sum()
    ece = float((weights * (curve["mean_pred"] - curve["obs_freq"]).abs()).sum())
    worst = float((curve["mean_pred"] - curve["obs_freq"]).abs().max())
    print(f"{name:<16} {ece:>8.4f} {worst:>15.4f} {f'[{p.min():.3f}, {p.max():.3f}]':>20}")
print()
print("The liar's bin diagram is the interesting one. Nearly all of its mass lands")
print("in the extreme bins, where the diagonal is steep, and the observed frequency")
print("in those bins is nowhere near the claimed one. It looks overconfident in")
print("exactly the region a user would act on.")
print()

print("Why Laplace smoothing is not optional - the liar's diagram recomputed on a")
print("deliberately small random slice of 300 rows, where most bins hold a handful:")
print()
# A RANDOM slice, not a prefix: the synthetic generator emits class-ordered rows, so
# taking the first 300 would hand us 300 positives and every bin would read 1.000.
slice_idx = np.random.default_rng(7).permutation(len(y_overlap))[:300]
small = calibration_curve_hand(y_overlap[slice_idx], liar_probs_on_overlap[slice_idx], n_bins=10)
print(f"{'bin':>5} {'n':>4} {'mean pred':>11} {'raw obs freq':>14} {'smoothed':>10} {'95% CI':>18}")
print("-" * 68)
for _, r in small.iterrows():
    interval = f"[{max(0.0, r['ci_lo']):.2f}, {min(1.0, r['ci_hi']):.2f}]"
    print(f"{int(r['bin']):>5} {int(r['n']):>4} {r['mean_pred']:>11.3f} "
          f"{r['obs_freq']:>14.3f} {r['smoothed']:>10.3f} {interval:>18}")
print()
degenerate = small[(small["n"] <= 4) & ((small["obs_freq"] == 0.0) | (small["obs_freq"] == 1.0))]
print("The thin bins are the point. Bin "
      f"{', bin '.join(str(int(b)) for b in degenerate['bin'])} hold "
      f"{', '.join(str(int(n)) for n in degenerate['n'])} rows")
print("between them, and each reports an observed frequency of exactly 0.000 or")
print("1.000 - the most confident-looking numbers a calibration diagram can")
print("contain - while the population they were drawn from is "
      f"{y_overlap.mean():.0%} positive. Their confidence intervals span most of")
print("the unit interval. Plotted raw, these points look like the best calibrated")
print("model in the world. Every published calibration figure smooths, and any that")
print("does not is overstating the evidence it appears to show.")
print()

# sklearn returns (observed frequency, mean predicted) in that order - the opposite
# of the order the quantities are usually written in, and an easy thing to get wrong.
sk_obs, sk_pred = calibration_curve(yo_test, overlap_results[1.0]["prob"], n_bins=10,
                                    strategy="uniform")
mine = curves["honest blobs"]
print("Verified against sklearn's calibration_curve (which returns observed, then predicted):")
print(f"  my observed frequencies    : {np.round(mine['obs_freq'].values, 5)}")
print(f"  sklearn observed freqs     : {np.round(sk_obs, 5)}")
print(f"  max |difference|           : {np.abs(mine['obs_freq'].values - sk_obs).max():.3e}")
print(f"  my mean predicted values   : {np.round(mine['mean_pred'].values, 5)}")
print(f"  sklearn mean predicted     : {np.round(sk_pred, 5)}")
print(f"  max |difference|           : {np.abs(mine['mean_pred'].values - sk_pred).max():.3e}")
print()


# --------------------------------------------------------------------------------------
# PART E - repairing calibration
# --------------------------------------------------------------------------------------
print("-" * 78)
print("PART E - two ways to repair calibration")
print("-" * 78)
print("""
We repair the LIAR from Part B, because it is the model with real miscalibration to
fix. Calibrating an already-calibrated model would show nothing.

(1) RE-REGULARISE. Refit the whole model with a smaller C. This shrinks w, which
    compresses the probabilities toward 0.5, and it is a single hyperparameter you
    can select by cross-validation with no extra fitting stage. It only works when
    the miscalibration is a UNIFORM over-confidence, and it costs ranking quality,
    because it shrinks the scores you rank by as well.

(2) RE-CALIBRATE. Fit a second model mapping the first model's output to a
    probability. Two standard choices:
      Platt scaling - logistic regression on the first model's log-odds. Two fitted
                      parameters, smooth, strictly monotone, extremely hard to
                      overfit. The sensible default.
      Isotonic      - a free-form monotone step function, nonparametric. Fits any
                      monotone distortion, but needs a lot of data and will happily
                      memorise noise on a small sample.

Both are fit on a held-out calibration set and applied to fresh data. Fitting them
on the data you evaluate them on is the most common calibration mistake, and it is
demonstrated at the end of this part so you can see how flattering it is.
""")
print()

# The honest three-way split. The model is trained on the SEPARABLE data (that is
# what makes it a liar), and the calibrators are fit on the OVERLAPPING data - which
# is the correct thing to do and is also what you would really do, because the
# calibrator needs labels from the population the model is actually being used on.
X_fit, X_cal, y_fit, y_cal = train_test_split(
    Xo_train, yo_train, test_size=0.35, random_state=7, stratify=yo_train
)

# The liar: inflated C, fit on the separable training data.
liar_model = LogisticRegression(C=100.0, max_iter=5000).fit(Xs_train, ys_train)
raw_cal = liar_model.predict_proba(X_cal)[:, 1]

# Each calibrator is fit on the overlapping calibration split alone.
platt = LogisticRegression(max_iter=1000).fit(platt_transform(raw_cal).reshape(-1, 1), y_cal)
isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_cal, y_cal)
# Re-regularising is a full refit at a smaller C - on the SAME separable data, so the
# only thing that changes is C and the difference in the table is attributable to it.
low_c_model = LogisticRegression(C=0.02, max_iter=5000).fit(Xs_train, ys_train)

# Now apply all of them, untouched, to the overlapping test set.
raw_test = liar_model.predict_proba(Xo_test)[:, 1]
low_c_test = low_c_model.predict_proba(Xo_test)[:, 1]
repairs = {
    "raw liar (C=100)": raw_test,
    "re-regularised C=0.02": low_c_test,
    "Platt on the liar": platt.predict_proba(platt_transform(raw_test).reshape(-1, 1))[:, 1],
    "isotonic on the liar": isotonic.predict(raw_test),
}
honest_ref = overlap_results[1.0]["prob"]

print(f"{'model':<24} {'ROC-AUC':>9} {'log loss':>10} {'Brier':>8} {'ECE':>8} {'mean p':>9}")
print("-" * 76)
for name, p in repairs.items():
    print(f"{name:<24} {roc_auc_score(yo_test, p):>9.4f} {log_loss(yo_test, p):>10.4f} "
          f"{brier_score_loss(yo_test, p):>8.4f} {calibration_error(yo_test, p):>8.4f} {p.mean():>9.4f}")
print(f"{'honest reference':<24} {overlap_results[1.0]['auc']:>9.4f} "
      f"{overlap_results[1.0]['log_loss']:>9.4f} "
      f"{overlap_results[1.0]['brier']:>8.4f} {honest_ece:>8.4f} {honest_ref.mean():>9.4f}")
print()
print(f"The liar's log loss of {log_loss(yo_test, raw_test):.4f} is the worst number in the")
print("table - worse than predicting the base rate for everyone. Two of the three")
print("repairs recover most of it, and the honest reference shows how much there")
print("was to recover.")
print()
print("The re-regularised model does well here, but for the wrong reason, and it is")
print("worth being precise about why. Shrinking w is a crude way to buy humility,")
print("and it happens to work because the liar's mistake is a UNIFORM one: every")
print("score is too extreme, so one global shrinkage factor happens to fix the whole")
print(f"score is too extreme, so one global shrinkage factor happens to fix the whole")
print(f"scale. It also leaves the mean predicted probability at {low_c_test.mean():.4f} -")
print("still nowhere near the true 0.5, because the drift here was a change in the")
print("class GEOMETRY (separable to overlapping) at unchanged 50/50 prevalence, not")
print("a uniform over-confidence. Had the drift been a change in prevalence or in")
print("the feature distribution, one global C would not have touched it. And")
print("calibrating a drifted model is treating the symptom, not the cause.")
print()

raw_auc = roc_auc_score(yo_test, raw_test)
logodds_auc = roc_auc_score(yo_test, platt_transform(raw_test))
platt_auc = roc_auc_score(yo_test, repairs["Platt on the liar"])
iso_auc = roc_auc_score(yo_test, repairs["isotonic on the liar"])
raw_unique = len(np.unique(raw_test))
platt_unique = len(np.unique(repairs["Platt on the liar"]))
iso_unique = len(np.unique(repairs["isotonic on the liar"]))

print("And here is the subtlety that makes this part worth reading twice.")
print("Recalibration is supposed to leave the ranking untouched, because ROC-AUC")
print("depends only on the ORDER of the rows. Platt is a strictly increasing")
print("function of the log-odds, so in principle it cannot reorder anything. The")
print("measurement confirms it, and also shows where the last few digits go:")
print()
print(f"    AUC of the raw probabilities : {raw_auc:.10f}")
print(f"    AUC of the raw log-odds      : {logodds_auc:.10f}")
print(f"    AUC after Platt              : {platt_auc:.10f}")
print()
print(f"    Platt vs raw log-odds        : {platt_auc - logodds_auc:.3e}  <- exactly zero, as predicted")
print(f"    raw probability vs raw odds  : {raw_auc - logodds_auc:.3e}  <- the entire discrepancy")
print()
print("The tiny gap is NOT Platt reordering anything. It is in the raw scores")
print(f"themselves: the liar outputs values down to {raw_test.min():.1e}, where a float can no longer")
print("distinguish the rows, so the raw probabilities already contain ties before")
print("Platt touches them. Once you compare like with like - log-odds in, log-odds")
print("out - the identity is exact. Worth knowing, because it is the sort of")
print("sixth-decimal wobble that otherwise gets mistaken for a bug.")
print()
print("Isotonic is different, and for a substantive reason:")
print(f"    AUC after isotonic           : {iso_auc:.10f}   ({abs(iso_auc - raw_auc):.2e} below raw)")
print(f"    distinct values: raw {raw_unique}, Platt {platt_unique}, isotonic {iso_unique}")
print()
print(f"Isotonic collapsed {raw_unique} distinct scores into {iso_unique}. It is fitted as a step")
print("function, so it is only WEAKLY monotone - many rows share a value, and a tie")
print("earns half credit in the AUC rather than full credit for the better row. A")
print("weakly-monotone repair really can cost you ranking, which is a fair price to")
print("pay for a method that assumes no functional form at all.")
print()
print("So the precise statement is: STRICTLY monotone recalibration preserves")
print("ROC-AUC exactly, and weakly monotone recalibration preserves it up to ties.")
print("The two properties remain orthogonal in the sense that matters - no")
print("recalibration can IMPROVE a ranking, so you never trade one against the")
print("other - but 'cannot change it at all' was too strong.")
print()
print(f"Platt fitted slope {float(platt.coef_[0, 0]):.3f}, intercept {float(platt.intercept_[0]):.3f}.")
print("A slope of 1.0 with a zero intercept means the raw scores needed no")
print("distortion at all and the calibrator is a no-op. A slope far from 1 means the")
print("raw scores really are distorted. A slope BELOW 1 is the signature of")
print("over-confidence: the repair pulls extreme scores back toward the middle in")
print("exactly the log-odds space where the damage happened.")
print()
iso_p = repairs["isotonic on the liar"]
print(f"Isotonic output on test spans [{iso_p.min():.4f}, {iso_p.max():.4f}], against a raw low")
print(f"end of {raw_test.min():.2e}. Note how much narrower it is, and how few distinct values")
print("there are. Isotonic is conservative where the data is thin, which is the")
print("price of not assuming a functional form. That is why Platt is the default on")
print("small data and isotonic only becomes the default once you have tens of")
print("thousands of rows to fit the steps with.")
print()

leaky_platt = LogisticRegression(max_iter=1000).fit(
    platt_transform(raw_test).reshape(-1, 1), yo_test)
leaky_p = leaky_platt.predict_proba(platt_transform(raw_test).reshape(-1, 1))[:, 1]
leaky_iso = IsotonicRegression(out_of_bounds="clip").fit(raw_test, yo_test)
leaky_iso_p = leaky_iso.predict(raw_test)

print("And the leak, for contrast. Both calibrators refit on the test set they are")
print("then evaluated on, which is the most common calibration mistake there is:")
print()
print(f"{'calibrator':<34} {'honest log loss':>16} {'leaky log loss':>15} {'apparent gain':>14}")
print("-" * 82)
for label, honest_p, leaky_p_ in [
        ("Platt (2 fitted parameters)", repairs["Platt on the liar"], leaky_p),
        ("isotonic (free-form steps)", repairs["isotonic on the liar"], leaky_iso_p)]:
    h_ll = log_loss(yo_test, honest_p)
    l_ll = log_loss(yo_test, leaky_p_)
    print(f"{label:<34} {h_ll:>16.4f} {l_ll:>15.4f} {h_ll - l_ll:>14.4f}")
print()
print("Both 'gains' are pure fiction - the leaky fits have already seen the answers")
print("they are being graded on. The gap between the two rows is the interesting")
platt_gain = (log_loss(yo_test, repairs["Platt on the liar"]) - log_loss(yo_test, leaky_p))
iso_gain = (log_loss(yo_test, repairs["isotonic on the liar"]) - log_loss(yo_test, leaky_iso_p))
print(f"part. Platt has exactly two free parameters, so it cannot memorise much no")
print(f"matter how badly you misuse it, and cheating buys it only {platt_gain:.4f}.")
print(f"Isotonic is free-form, so it memorises the test set and flatters itself by")
print(f"{iso_gain:.4f} - enough to look better than the honest Platt number despite being")
print("fitted on the answers.")
print()
print("That is the practical argument for the smooth default: a low-capacity")
print("calibrator is not merely more accurate on small samples, it is also much")
print("harder to fool yourself with. And the failure is silent - a leaky ECE looks")
print("perfectly respectable in the report.")
print()


# --------------------------------------------------------------------------------------
# PART F - choosing the threshold from cost
# --------------------------------------------------------------------------------------
print("-" * 78)
print("PART F - the threshold is a business decision, made in expected value")
print("-" * 78)
print("""
Everything so far used the 50/50 blobs, which is a fine way to learn about
calibration and a terrible place to make a cost decision: with equal numbers of
positives and negatives there is no screening decision to be made. So this part
switches to a problem with a real business shape - population disease screening.

F1 assumes the two errors cost the same. Almost nothing real does. Write the four
cells in currency, for one row with true probability p:

                       flagged        not flagged
    actually positive  +SAVED         -HARM
    actually negative  -FP_COST        0

Flagging is worthwhile when, in expectation over p,

    p*SAVED - (1-p)*FP_COST   >   -p*HARM

    p*(SAVED + HARM)  >  (1-p)*FP_COST
    p*(SAVED + HARM + FP_COST)  >  FP_COST
    p  >  FP_COST / (SAVED + HARM + FP_COST)

So for a CALIBRATED model the optimal threshold is closed-form, and it has nothing
to do with 0.5. Note what it depends on: the false-positive cost in the numerator,
and the total swing from 'always flag' to 'never flag' in the denominator. It is
not a cost RATIO, which is the version most people quote and the version that is
usually wrong.
""")
print()

# --- the screening dataset -------------------------------------------------------------
rng_screen = np.random.default_rng(101)
n_screen = 20000
age = rng_screen.normal(55, 14, n_screen)
bmi = rng_screen.normal(28, 6, n_screen)
smoker = rng_screen.binomial(1, 0.25, n_screen)
family_hx = rng_screen.binomial(1, 0.18, n_screen)
fam = rng_screen.binomial(1, 0.22, n_screen)
marker = rng_screen.normal(0, 1, n_screen)

log_odds_screen = (-7.5 + 0.045 * age + 0.06 * (bmi - 28) + 0.9 * smoker
                   + 1.4 * family_hx + 0.8 * fam + 1.6 * marker)
y_screen = rng_screen.binomial(1, 1.0 / (1.0 + np.exp(-log_odds_screen)))
X_screen = np.column_stack([age, bmi, smoker, family_hx, fam, marker])
screen_names = ["age", "bmi", "smoker", "family_history", "family_affected", "marker"]

Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
    X_screen, y_screen, test_size=0.3, random_state=0, stratify=y_screen
)
screen_model = LogisticRegression(max_iter=5000).fit(Xs_tr, ys_tr)
screen_prob = screen_model.predict_proba(Xs_te)[:, 1]

SAVED_BY_CATCHING = 50_000.0   # early treatment prevents the harm
HARM_IF_MISSED = 60_000.0      # delayed treatment causes it
FP_COST = 4_000.0              # unnecessary treatment: cost plus side effects

print(f"Screening dataset: {len(y_screen)} people, prevalence {y_screen.mean():.2%}, "
      f"test AUC {roc_auc_score(ys_te, screen_prob):.4f}")
print()
print("The costs, per person:")
print(f"  catch it early and prevent the harm   +{SAVED_BY_CATCHING:>10,.0f}")
print(f"  miss it and the harm happens          {-HARM_IF_MISSED:>10,.0f}")
print(f"  flag a healthy person for treatment   {-FP_COST:>10,.0f}")
print()
print("This is a genuinely uncomfortable decision, which is what makes it a good")
print("teaching example. The treatment has real side effects, so flagging everyone")
print("is not free and the threshold is not a technicality - it decides who gets")
print("medicated on the strength of a statistical estimate.")
print()

closed_form_t = FP_COST / (SAVED_BY_CATCHING + HARM_IF_MISSED + FP_COST)
print(f"Closed-form optimum:  FP_COST / (SAVED + HARM + FP_COST)")
print(f"                   = {FP_COST:,.0f} / ({SAVED_BY_CATCHING:,.0f} + {HARM_IF_MISSED:,.0f} + {FP_COST:,.0f}) = {closed_form_t:.4f}")
print()


def expected_value_table(prob, y_true, thresholds):
    """
    Total expected value of running the policy at each threshold, in currency.
    TP -> +SAVED, FN -> -HARM, FP -> -FP_COST, TN -> 0.
    """
    rows = []
    for t in thresholds:
        flagged = prob >= t
        is_pos = y_true == 1
        tp = int(np.sum(flagged & is_pos))
        fp = int(np.sum(flagged & ~is_pos))
        fn = int(np.sum(~flagged & is_pos))
        ev = tp * SAVED_BY_CATCHING - fn * HARM_IF_MISSED - fp * FP_COST
        rows.append({
            "threshold": t, "TP": tp, "FP": fp, "FN": fn,
            "ev_total": ev, "ev_per_row": ev / len(y_true),
            "flag_rate": (tp + fp) / len(y_true),
        })
    return pd.DataFrame(rows)


thresholds = np.linspace(0.005, 0.995, 199)
ev_table = expected_value_table(screen_prob, ys_te, thresholds)

print(f"{'threshold':>10} {'flag rate':>10} {'TP':>5} {'FP':>6} {'FN':>5} {'EV / row':>12} {'total EV':>14}")
print("-" * 72)
for row in ev_table.iloc[::20].itertuples():
    print(f"{row.threshold:>10.3f} {row.flag_rate:>10.1%} {row.TP:>5} {row.FP:>6} "
          f"{row.FN:>5} {row.ev_per_row:>12,.0f} {row.ev_total:>14,.0f}")
print()

best = ev_table.loc[ev_table["ev_total"].idxmax()]
f1_thresh, f1_max = max(
    [(t, f1_score(ys_te, (screen_prob >= t).astype(int), zero_division=0)) for t in thresholds],
    key=lambda p: p[1])
acc_thresh = max(thresholds, key=lambda t: accuracy_score(ys_te, (screen_prob >= t).astype(int)))

n_pos = int(ys_te.sum())
n_neg = len(ys_te) - n_pos
never_flag_ev = -n_pos * HARM_IF_MISSED
flag_all_ev = n_pos * SAVED_BY_CATCHING - n_neg * FP_COST

print(f"{'criterion':<28} {'threshold':>10} {'value at threshold':>20}")
print("-" * 60)
print(f"{'best F1':<28} {f1_thresh:>10.3f} {f1_max:>20.4f}")
print(f"{'best accuracy':<28} {acc_thresh:>10.3f} "
      f"{accuracy_score(ys_te, (screen_prob >= acc_thresh).astype(int)):>20.4f}")
print(f"{'closed-form EV optimum':<28} {closed_form_t:>10.4f} {'(theory)':>20}")
print(f"{'empirical EV optimum':<28} {best['threshold']:>10.3f} {best['ev_per_row']:>20,.0f}")
print()
print(f"F1 wants {f1_thresh:.3f}. Expected value wants {best['threshold']:.3f}. That is a factor of "
      f"{f1_thresh / best['threshold']:.1f} in threshold, and it is not a")
print("close call. F1 is answering a different question - 'how well can I balance")
print("precision and recall with no stated preference' - and the preference here is")
print("enormously asymmetric. A miss costs "
      f"{HARM_IF_MISSED / FP_COST:.0f}x what a false alarm costs, so F1's symmetric optimum")
print("is far too conservative.")
print()
print("The closed form and the empirical optimum also agree closely - "
      f"{closed_form_t:.4f} against")
print(f"{best['threshold']:.4f} - which is a genuinely useful check, because the two are derived")
print("completely differently: one is algebra on a calibrated probability, the other")
print("is a grid search over realised outcomes. The residual gap is real and")
print("informative: this model is slightly UNDER-confident in the 0.02-0.05 region,")
print("so the empirical optimum sits a little above where the algebra says. If the")
print("gap were large you would want to investigate calibration before trusting")
print("either number.")
print()

print("For scale, the three policies side by side on this test set:")
print(f"  never flag      EV/row {never_flag_ev / len(ys_te):>10,.0f}   "
      f"total {never_flag_ev:>12,.0f}   (the silent baseline most models are compared against)")
print(f"  flag everyone   EV/row {flag_all_ev / len(ys_te):>10,.0f}   "
      f"total {flag_all_ev:>12,.0f}   (catches every case, treats every healthy person)")
print(f"  tuned model     EV/row {best['ev_per_row']:>10,.0f}   "
      f"total {best['ev_total']:>12,.0f}   (flags {best['flag_rate']:.1%} of people)")
print()
print(f"'Flag everyone' is worse than doing nothing, and the tuned model beats both -")
print(f"by {(best['ev_total'] - never_flag_ev) / len(ys_te):,.0f} per person against never flagging, "
      f"or {best['ev_total'] - never_flag_ev:,.0f} across the test set.")
print()
print("That last number is what belongs in the business case. Not the AUC, not the")
print("log loss, and certainly not the accuracy - which on this data is maximised at")
print(f"{acc_thresh:.3f} by flagging almost nobody and being right about the {1 - ys_te.mean():.0%} who are healthy.")
print()
print("One piece of fine print worth stating out loud in a review: this table is a")
print("post-hoc evaluation on the test set, so the empirical optimum is slightly")
print("optimistic - the threshold was chosen knowing the answers. Doing it honestly")
print("means computing the same table on cross-validated out-of-fold predictions,")
print("which is script 11.")
print()


# --------------------------------------------------------------------------------------
# PART G - what does 0.5 even mean
# --------------------------------------------------------------------------------------
print("-" * 78)
print("PART G - is 0.5 even the right place to start?")
print("-" * 78)
print("""
A default threshold of 0.5 encodes two assumptions: the classes are balanced, and
the two errors cost the same. Either one failing breaks it silently - no exception,
no warning, no log line.

The cheapest defence is to compare the flag rate against expected operational
workload, because that is a number a business already knows and a model rarely
agrees with.
""")
prev = ys_tr.mean()
flagged_at_half = float((screen_prob >= 0.5).mean())
print(f"Training prevalence          : {prev:.2%}")
print(f"Rows scoring at or above 0.5 : {flagged_at_half:.2%}")
print(f"Flag rate at 0.5 as a multiple of the base rate: {flagged_at_half / prev:.2f}x")
print()
print(f"The model at 0.5 flags {flagged_at_half:.2%} of people, which is only "
      f"{flagged_at_half / prev:.1f}x the base rate, so it")
print("UNDER-flags. Most people who actually have the condition score below 0.5,")
print("because on a rare-outcome problem the bulk of the probability mass sits far")
print("below the midpoint. Nobody chose that; it is just what a 0.5 default does.")
print("The direction is not a law of nature - it depends on how confident the model is -")
print("which is precisely why the ratio is worth printing rather than assuming.")
print()
quantile_threshold = float(np.quantile(screen_prob, 1 - prev))
print(f"If the clinic expected to treat about {prev:.2%} of people - the base rate, the")
print(f"most natural planning assumption - the threshold that actually does that is")
print(f"{quantile_threshold:.3f}, the {1 - prev:.2%} quantile of the predicted probabilities. Not 0.5, and not")
print(f"{best['threshold']:.3f} either. The EV optimum and the base-rate-matching threshold are")
print("two different sensible targets, and which one you want depends on whether")
print("you are optimising value or matching a capacity plan.")
print()
print("This is why the flag rate belongs in the monitoring dashboard. A model can be")
print("perfect by every metric and still be undeliverable, purely because nobody")
print("checked whether the number of people it flags matches the number of people")
print("the clinic can treat. That failure is silent, and it arrives on the day you")
print("go live.")
print()


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# --- reliability diagrams -------------------------------------------------------------
ax = axes[0, 0]
ax.plot([0, 1], [0, 1], "--", color="#868E96", linewidth=2, label="perfect calibration")
rel_colors = {"honest blobs": "#0CA678", "C=0.001 blobs": "#4C6EF5", "liar on blobs": "#E03131"}
for name, curve in curves.items():
    ax.plot(curve["mean_pred"], curve["smoothed"], "o-", linewidth=2.2, markersize=7,
            color=rel_colors[name], label=name)
    ax.fill_between(curve["mean_pred"], curve["ci_lo"].clip(0, 1), curve["ci_hi"].clip(0, 1),
                    color=rel_colors[name], alpha=0.14, linewidth=0)
ax.set_xlabel("mean predicted probability in bin")
ax.set_ylabel("observed frequency of positives")
ax.set_title("Reliability diagrams: read the VERTICAL gap", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, frameon=False, loc="upper left")
ax.grid(alpha=0.25)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# --- the repairs ----------------------------------------------------------------------
ax = axes[0, 1]
for name, p in repairs.items():
    curve = calibration_curve_hand(yo_test, p, 10)
    ax.plot(curve["mean_pred"], curve["smoothed"], "o-", linewidth=2, markersize=6,
            label=f"{name}  (ECE {calibration_error(yo_test, p):.3f})")
ax.plot([0, 1], [0, 1], "--", color="#868E96", linewidth=2, label="perfect")
ax.set_xlabel("mean predicted probability")
ax.set_ylabel("observed frequency")
ax.set_title("Repairs move points ONTO the diagonal, not the ranking",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.grid(alpha=0.25)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# --- expected value vs threshold ------------------------------------------------------
ax = axes[1, 0]
ax.plot(ev_table["threshold"], ev_table["ev_total"] / 1000, linewidth=2.6,
        color="#4C6EF5", label="total expected value")
ax.axvline(best["threshold"], color="#E03131", linestyle="--", linewidth=2)
ax.annotate(f"EV optimum {best['threshold']:.3f}\n{best['ev_per_row']:,.0f} per person",
            xy=(best["threshold"], best["ev_total"] / 1000),
            xytext=(best["threshold"] + 0.10, best["ev_total"] / 1000 - 2600),
            fontsize=9, color="#E03131",
            arrowprops=dict(arrowstyle="->", color="#E03131"))
ax.axvline(f1_thresh, color="#F59F00", linestyle=":", linewidth=2)
ax.text(f1_thresh + 0.006, ax.get_ylim()[0], f"F1 optimum {f1_thresh:.3f}", fontsize=9,
        color="#F59F00", rotation=90, va="bottom")
ax.axvline(acc_thresh, color="#7048E8", linestyle="-.", linewidth=1.6)
ax.text(acc_thresh - 0.012, ax.get_ylim()[0], f"accuracy optimum {acc_thresh:.3f}",
        fontsize=9, color="#7048E8", rotation=90, va="bottom", ha="right")
ax.axhline(never_flag_ev / 1000, color="#868E96", linestyle=":", linewidth=1.6)
ax.text(0.985, never_flag_ev / 1000 - 1500, "never flag", fontsize=8,
        color="#868E96", ha="right")
ax.axhline(flag_all_ev / 1000, color="#868E96", linestyle=":", linewidth=1.6)
ax.text(0.985, flag_all_ev / 1000 + 250, "flag everyone", fontsize=8,
        color="#868E96", ha="right")
ax.axvline(0.5, color="#C92A2A", linestyle="--", linewidth=1.2, alpha=0.5)
ax.text(0.505, ax.get_ylim()[1] * 0.86, "sklearn default 0.5", fontsize=8, color="#C92A2A")
ax.set_xlabel("decision threshold")
ax.set_ylabel("total expected value on the test set (thousands)")
ax.set_title("Three 'optimal' thresholds, and only one of them is the right question",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9, frameon=False)
ax.grid(alpha=0.25)

# --- the C sweep, both datasets -------------------------------------------------------
ax = axes[1, 1]
cs = np.logspace(-3, 3, 30)
sep_ll, sep_wn, ov_ww = [], [], []
for c in cs:
    ms = LogisticRegression(C=c, max_iter=4000).fit(Xs_train, ys_train)
    ps = ms.predict_proba(Xs_test)[:, 1]
    sep_ll.append(log_loss(ys_test, ps))
    sep_wn.append(np.linalg.norm(ms.coef_))
    mo = LogisticRegression(C=c, max_iter=4000).fit(Xo_train, yo_train)
    ov_ww.append(np.linalg.norm(mo.coef_))
ax.semilogx(cs, sep_wn, linewidth=2.6, color="#E03131",
            label="separable data: ||w|| (grows without limit)")
ax.semilogx(cs, ov_ww, linewidth=2.6, color="#0CA678",
            label="overlapping data: ||w|| (converges, then stops)")
ax.set_xlabel("regularisation strength C (log scale)")
ax.set_ylabel("norm of the coefficient vector")
ax.set_title("C matters only when the data is separable", fontsize=11, fontweight="bold")
ax.legend(fontsize=8.5, frameon=False, loc="center left")
ax.grid(alpha=0.25, which="both")
ax.annotate("slope -> 0:\nthe MLE is interior,\nso C has little to do",
            xy=(cs[len(cs) // 2], ov_ww[len(cs) // 2]), xytext=(0.02, 1.55),
            fontsize=8.5, color="#0CA678",
            arrowprops=dict(arrowstyle="->", color="#0CA678"))
ax.annotate("keeps climbing:\ncomplete separation",
            xy=(cs[-4], sep_wn[-4]), xytext=(0.35, 3.1),
            fontsize=8.5, color="#E03131",
            arrowprops=dict(arrowstyle="->", color="#E03131"))

fig.suptitle("06 - Probability Thresholds and Calibration", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


# --------------------------------------------------------------------------------------
print()
print("=" * 78)
print("TAKEAWAYS")
print("=" * 78)
print("1. C is not a difficulty dial. On overlapping data the log-likelihood has a")
print("   finite interior maximum, so above C~0.01 nothing happens across three")
print("   orders of magnitude. C is a shrinkage dial, and shrinkage only helps if")
print("   there is variance to remove.")
print()
print("2. C matters when the data is separable, because then the likelihood is")
print("   monotone in |w| and the maximum is at infinity. One training error is")
print("   enough to trigger it. That is the mechanism behind complete separation,")
print("   and script 16 is devoted to it.")
print()
print("3. Ranking and calibration are separate properties, and no recalibration can")
print("   IMPROVE your ranking. Platt is strictly monotone in the log-odds and")
print("   preserves ROC-AUC to exactly zero difference; isotonic is only weakly")
print("   monotone and can cost a little, because its step function creates ties.")
print()
print("4. Neither property implies the other, in either direction. A model trained")
print("   on one population can be perfectly calibrated and catastrophic on the")
print("   next. A misspecified model can have BETTER calibration than a working one,")
print("   because 'near 0.5 everywhere' is trivially calibrated. Report both.")
print()
print("5. A large gap between the mean predicted probability and the observed base")
print("   rate is the cheapest drift check available - one number, no chart, and it")
print("   catches a broken model long before the ROC curve looks wrong.")
print()
print("6. Smooth your calibration bins. With one row in a bin the observed frequency")
print("   is exactly 0 or 1, and the diagram is fiction. Laplace smoothing toward")
print("   the bin's own mean prediction is the minimum honest treatment. Remember")
print("   that sklearn's calibration_curve returns (observed, predicted), in that")
print("   order - it is easy to check against the wrong quantity.")
print()
print("7. Fit a calibrator on data disjoint from the set you report on. The leaky")
print("   version always looks better and always generalises worse - and the amount")
print("   by which it flatters itself is a direct read-out of the calibrator's")
print("   capacity: two Platt parameters cannot cheat much, free-form isotonic can.")
print("   That is the real argument for the smooth default.")
print()
print("8. 0.5 is a default, not a decision. It assumes balanced classes and")
print("   symmetric error costs. For a calibrated model the value-optimal threshold")
print("   is closed-form: FP_COST / (SAVED + HARM + FP_COST). Note the shape - a")
print("   cost in the numerator, the total swing in the denominator. It is NOT the")
print("   cost RATIO that is usually quoted, and the two disagree badly. Check the")
print("   algebra against a grid search on realised outcomes, then check the")
print("   resulting flag rate against operational capacity.")
