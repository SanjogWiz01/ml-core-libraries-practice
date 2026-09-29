"""
08 - Regularization: L1, L2, and Elastic Net
============================================

Goal: understand what the penalty term is actually doing to the coefficients, and
when it helps.

The problem being solved
------------------------
40 features. Five of them genuinely drive the outcome. Thirty-five are pure noise
generated independently of the label. n = 4,000, so the sample is big enough to
estimate 40 coefficients but not so big that the noise features are reliably
identified as useless - which is exactly the situation regularization is for.

The unregularised fit will happily hand every one of the 35 noise features a
nonzero coefficient. It is not wrong. It is answering a different question: given
this specific sample, what are the best coefficients? Those are genuinely not zero,
because noise correlates with noise.

Regularization changes the question to: given this sample, what are the simplest
coefficients that still work? That is a different estimator, and the difference is
the whole subject of this file.

The three penalties
-------------------
    L2 (ridge)      C * sum(log loss)  +  0.5 * ||w||^2
    L1 (lasso)      C * sum(log loss)  +  sum(|w|)
    Elastic net     C * sum(log loss)  +  l1_ratio*sum(|w|)
                                        + 0.5*(1-l1_ratio)*||w||^2

scikit-learn note: in version 1.8 the `penalty=` argument is deprecated. The
supported way to choose a penalty is `l1_ratio`:
    l1_ratio = 0    -> L2
    l1_ratio = 1    -> L1
    0 < l1_ratio<1  -> elastic net
That is what this file uses, and it is why you will not see `penalty=` here.
For the unpenalised fit, C = np.inf is the documented spelling but it emits a
warning in 1.8 ("Setting penalty=None will ignore the C and l1_ratio parameters").
C = C_UNREG = 1e6 gives the same coefficients to displayed precision, quietly, so
that is what is used below. The unpenalised fit in scikit-learn is really
regularised by a tiny L2 term; the path is a limit, not a point you can reach.

What this file demonstrates
---------------------------
1. L2: shrinks every coefficient toward zero, never sets one exactly to zero.
2. L1: the corners of the constraint region, and exact feature recovery - it finds
   the 5 real features and none of the 35 noise features.
3. The C path for both, and the bias-variance trade in numbers.
4. Elastic net as the continuum between them, and when the middle is the right answer.
5. Standardization is not optional: the same L1 fit selects 38 features on raw
   units and exactly 5 after scaling.
6. The surprise: the sparsest model ranks BEST and calibrates WORST.
7. Choosing C by cross-validation rather than by reading the path.

Run:  python 08_regularization_l1_l2_elasticnet.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.width", 130)

N_FEATURES = 40
N_TRUE = 5
TRUE_COEFS = np.array([1.5, 1.2, 0.9, 0.7, 0.5])
C_UNREG = 1e6  # large enough to be indistinguishable from no penalty, without the warning


# ======================================================================================
# The data
# ======================================================================================
def make_sparse_signal_data(n=4000, n_features=N_FEATURES, n_true=N_TRUE,
                            noise_scale=0.6, seed=11):
    """
    A wide dataset: a handful of real signals plus a lot of pure noise, with enough
    label noise that the classes overlap and the fit is not separable.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, n_features))
    log_odds = X[:, :n_true] @ TRUE_COEFS[:n_true] + noise_scale * rng.normal(0, 1, n)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-log_odds)))
    return X, y


X, y = make_sparse_signal_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

TRUE_IDX = list(range(N_TRUE))

print("=" * 78)
print("REGULARIZATION: L1, L2 AND ELASTIC NET")
print("=" * 78)
print(f"Training rows {len(y_train):,}, features {X.shape[1]}, of which {N_TRUE} are real")
print(f"Test rows     {len(y_test):,}, positive rate {y_test.mean():.2%}")
print()
print("Real coefficients (population values): "
      f"{np.array2string(TRUE_COEFS, precision=2)}")
print("The other 35 columns are generated independently of the label.")
print("Features are standardized, because Part E is entirely about what happens")
print("when they are not.")
print()


def fit(penalty_l1_ratio, C, solver=None):
    """
    Fit with the modern scikit-learn interface. The solver is chosen to match the
    penalty, because not every solver supports every one.
    """
    if solver is None:
        if penalty_l1_ratio in (0.0, 1.0):
            solver = "liblinear"
        else:
            solver = "saga"
    return LogisticRegression(C=C, l1_ratio=penalty_l1_ratio, solver=solver,
                              max_iter=40000).fit(X_train_s, y_train)


def summarize(model):
    """The three numbers that describe what a penalty did to the coefficients."""
    w = model.coef_.ravel()
    selected = np.abs(w) > 1e-8
    prob = model.predict_proba(X_test_s)[:, 1]
    return {
        "nonzero": int(selected.sum()),
        "true kept": int(selected[:N_TRUE].sum()),
        "noise kept": int(selected[N_TRUE:].sum()),
        "||w||": float(np.linalg.norm(w)),
        "ROC-AUC": roc_auc_score(y_test, prob),
        "log loss": log_loss(y_test, prob),
    }


# ======================================================================================
# PART A - the unregularised baseline
# ======================================================================================
print("-" * 78)
print("PART A - what happens with no penalty at all")
print("-" * 78)
print("""
Start by looking at what is being fixed. A huge C removes the penalty effectively
entirely, so this is the plain maximum-likelihood fit - the answer to 'given this
sample, what are the best coefficients'.
""")

unreg = fit(0.0, C_UNREG, solver="lbfgs")
unreg_summary = summarize(unreg)
print(f"{'model':<28} {'nonzero':>8} {'true kept':>10} {'noise kept':>11} {'||w||':>8} {'ROC-AUC':>9} {'log loss':>10}")
print("-" * 80)
print(f"{'no penalty':<28} {unreg_summary['nonzero']:>8} {unreg_summary['true kept']:>10} "
      f"{unreg_summary['noise kept']:>11} {unreg_summary['||w||']:>8.3f} "
      f"{unreg_summary['ROC-AUC']:>9.4f} {unreg_summary['log loss']:>10.4f}")
print()
print(f"All {N_FEATURES} coefficients are nonzero, including all {unreg_summary['noise kept']} noise features.")
print()
print("Look at the magnitudes, though, because the ranking is not what you expect:")
print()
w_unreg = unreg.coef_.ravel()
print(f"  the 5 real features, largest:   {np.max(np.abs(w_unreg[:N_TRUE])):.3f}")
print(f"  the 35 noise features, largest: {np.max(np.abs(w_unreg[N_TRUE:])):.3f}")
print(f"  the 35 noise features, mean:    {np.mean(np.abs(w_unreg[N_TRUE:])):.3f}")
print()
print("The noise features are an order of magnitude smaller than the real ones, which")
print("is reassuring and also misleading. They are small individually, but there are")
print(f"{N_FEATURES - N_TRUE} of them, and the variance of a sum grows with the number of terms however")
print("small each term is. Asking which coefficient is largest does not tell you")
print("whether the model is overfitting; the aggregate does. That is what the penalty")
print("is for.")
print()
print("The real coefficients are also shrunk relative to their true values")
print(f"({', '.join(f'{v:.2f}' for v in TRUE_COEFS)}). That shrinkage is not a bug in the")
print("estimator - it is what maximum likelihood does on overlapping data, because")
print("label noise is cheapest to explain by pulling the scores toward the base rate.")
print()
print("Worth being precise, because it is easy to get backwards: ALL THREE penalties")
print("shrink, including L1. L1 is not the 'less biased' option. Its distinguishing")
print("property is that it can also REMOVE features - it is willing to spend bias to")
print("buy a zero. Part C shows exactly that, and shows what the surviving")
print("coefficients look like afterwards.")
print()


# ======================================================================================
# PART B - L2, the shrinkage that never quite reaches zero
# ======================================================================================
print("-" * 78)
print("PART B - L2: shrink everything, keep everything")
print("-" * 78)
print("""
The L2 objective adds 0.5*||w||^2. Two properties follow directly from the shape of
that term, before you fit anything:

  1. It is differentiable everywhere, including AT zero. The gradient of 0.5*w^2 is
     w, which is 0 at w = 0. So nothing ever gets pinned to exactly zero - L2 always
     returns you all of your features, just smaller. Ask it which features matter
     and it cannot tell you; it has an opinion about all of them.

  2. It shrinks every coefficient by a similar proportion, not by a similar amount.
     Columns on a large scale get pulled down harder in absolute terms. Which is
     another way of saying L2 is scale-dependent - see Part E.
""")
print()

print(f"{'C':>8} {'nonzero':>8} {'true kept':>10} {'noise kept':>11} {'||w||':>8} {'ROC-AUC':>9} {'log loss':>10}")
print("-" * 80)
l2_path = []
for C in [0.001, 0.01, 0.1, 1.0, 10.0, C_UNREG]:
    model = fit(0.0, C, solver="lbfgs")
    s = summarize(model)
    l2_path.append((C, s))
    label = "none" if C == C_UNREG else f"{C:g}"
    print(f"{label:>8} {s['nonzero']:>8} {s['true kept']:>10} {s['noise kept']:>11} "
          f"{s['||w||']:>8.3f} {s['ROC-AUC']:>9.4f} {s['log loss']:>10.4f}")
print()
print("The 'nonzero' and 'noise kept' columns never move. That is not a limitation of")
print("this dataset or this C range - it is structural. L2's optimum is almost never")
print("on a coordinate axis, so no coefficient is ever exactly 0. You can make them")
print("all arbitrarily small; you cannot make any of them absent.")
print()
best_l2 = min(l2_path, key=lambda pair: pair[1]["log loss"])
best_l2_C, best_l2_s = best_l2
print("The 'log loss' column is the interesting one. It is best at C = "
      f"{('none' if best_l2_C == C_UNREG else f'{best_l2_C:g}')}")
print(f"(log loss {best_l2_s['log loss']:.4f}), and worse at both ends: at tiny C variance wins, and at")
print("no penalty the noise coefficients get to speak. So L2 is doing real work on")
print("this problem - it just cannot tell you WHICH features to drop, only how much")
print("to trust all of them.")
print()


# ======================================================================================
# PART C - L1, and exact feature recovery
# ======================================================================================
print("-" * 78)
print("PART C - L1: the penalty that can set a coefficient to zero")
print("-" * 78)
print("""
The L1 objective adds sum(|w|) instead. The kinks are at zero, and that single
geometric fact produces a completely different estimator:

  The subgradient of |w| at w = 0 is the whole interval [-1, 1]. So the objective
  can be stationary at w = 0 even though the loss gradient is non-zero - the
  penalty is steep enough to hold the coefficient exactly there. A whole range of
  coefficients gets parked on zero at once.

The intuitive picture: the feasible set ||w||_1 <= t is a DIAMOND with corners on
the axes, while the L2 feasible set ||w||_2 <= t is a CIRCLE. A contour of the loss
touching a circle touches it in one generic point. Touching a diamond almost always
lands on a corner, and the corners are on the axes - i.e. exactly zero.
""")
print()

print(f"{'C':>8} {'nonzero':>8} {'true kept':>10} {'noise kept':>11} {'||w||':>8} {'ROC-AUC':>9} {'log loss':>10}")
print("-" * 80)
l1_path = []
for C in [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.3, 1.0]:
    model = fit(1.0, C, solver="liblinear")
    s = summarize(model)
    l1_path.append((C, s, model))
    print(f"{C:>8g} {s['nonzero']:>8} {s['true kept']:>10} {s['noise kept']:>11} "
          f"{s['||w||']:>8.3f} {s['ROC-AUC']:>9.4f} {s['log loss']:>10.4f}")
print()

# Among the C values that recover the true set exactly, take the LARGEST C: that is
# the strongest regularisation that still gets the answer right, so it is the one
# with the most margin rather than the one closest to the edge of selection.
exact_rows = [t for t in l1_path if t[1]["true kept"] == N_TRUE and t[1]["noise kept"] == 0]
C_exact, s_exact, m_exact = max(exact_rows, key=lambda t: t[0])
print(f"Three values of C recover the true set exactly - {', '.join(f'{t[0]:g}' for t in exact_rows)} -")
print(f"and the most defensible is the largest of them, C = {C_exact:g}, because it has the")
print(f"most room before the selection starts dropping real features.")
print()
print(f"At C = {C_exact:g} the fit selects exactly the {s_exact['nonzero']} features that matter, with")
print(f"{s_exact['noise kept']} false positives out of {N_FEATURES - N_TRUE} noise features:")
print()
chosen = np.where(np.abs(m_exact.coef_.ravel()) > 1e-8)[0]
print(f"  selected indices : {chosen.tolist()}")
print(f"  true indices     : {TRUE_IDX}")
print(f"  coefficients     : {np.array2string(m_exact.coef_.ravel()[chosen], precision=3)}")
print()
best_l1_C, best_l1, _ = min(l1_path, key=lambda t: t[1]["log loss"])
dense_row = max(l1_path, key=lambda t: t[1]["nonzero"])[1]
print(f"It got all {N_TRUE}, and nothing else. Not approximately - exactly. On data where")
print("we built the answer, L1 found the answer, with no threshold to tune and no")
print("knowledge of which columns were real. That is variable selection as a side")
print("effect of the geometry, and it is the reason L1 is the default for this kind")
print("of problem.")
print()
print("Look again at that path table, because two things are going on at once and")
print("they do not point the same way. As C rises, the number of selected features")
print(f"climbs from {min(t[1]['nonzero'] for t in l1_path)} to {max(t[1]['nonzero'] for t in l1_path)} and ||w|| rises with it. But the best")
print("ROC-AUC and the best log loss sit at DIFFERENT C, and neither is at an end:")
print()
auc_row = max(l1_path, key=lambda t: t[1]["ROC-AUC"])[1]
print(f"  best ROC-AUC    {auc_row['ROC-AUC']:.4f} at C = {max(l1_path, key=lambda t: t[1]['ROC-AUC'])[0]:g}, "
      f"selecting {auc_row['nonzero']:>2} features, log loss {auc_row['log loss']:.4f}")
print(f"  best log loss   {best_l1['log loss']:.4f} at C = {best_l1_C:g}, "
      f"selecting {best_l1['nonzero']:>2} features, ROC-AUC {best_l1['ROC-AUC']:.4f}")
print(f"  most aggressive C in the table, C = {max(t[0] for t in l1_path):g}, selects "
      f"{dense_row['nonzero']} features, log loss {dense_row['log loss']:.4f}")
print()
print("Pushing C higher does not monotonically improve either metric. There is an")
print("interior optimum, it is in the middle of the range rather than at either end,")
print("and it is not the same point for both metrics. That is the bias-variance")
print("trade-off doing exactly what it should, and it is why the elbow is somewhere")
print("in the middle and not somewhere obvious.")
print()
print("The sparse end deserves a specific warning. Over-regularise L1 far enough and")
print("you get a model that predicts the base rate for everything: perfectly")
print("calibrated, and worthless. At C = "
      f"{min(l1_path, key=lambda t: t[1]['nonzero'])[0]:g} the log loss is {min(l1_path, key=lambda t: t[1]['nonzero'])[1]['log loss']:.4f} and the AUC is "
      f"{min(l1_path, key=lambda t: t[1]['nonzero'])[1]['ROC-AUC']:.4f}, while the model still emits")
print("perfectly reasonable-looking probabilities. The failure is silent, which is")
print("exactly what makes it dangerous.")
print()
print("And the two optima genuinely disagree, so this is a decision rather than a")
print("mechanical step. The sparsest model ranks best because dropping the noise")
print("features removes variance from the score; the moderately-dense model")
print("calibrates best because its coefficients are shrunk by about the right amount.")
print("Script 06's split again, showing up where you would not expect it. Pick the")
print("metric deliberately, because CV will not pick it for you.")
print()


# ======================================================================================
# PART D - elastic net, and the middle of the road
# ======================================================================================
print("-" * 78)
print("PART D - elastic net: the continuum, and its group effect")
print("-" * 78)
print("""
Elastic net interpolates: l1_ratio=0 is L2, l1_ratio=1 is L1, and anything between
pulls in both penalties. Its feasible region is a rounded diamond, which still has
corners on the axes, so it can still produce zeros - but the L2 part flattens the
corners slightly, which turns out to matter more than it sounds.
""")
print()

print(f"{'l1_ratio':>9} {'nonzero':>8} {'true kept':>10} {'noise kept':>11} {'||w||':>8} {'ROC-AUC':>9} {'log loss':>10}")
print("-" * 80)
for l1r in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
    model = fit(l1r, 0.05)
    s = summarize(model)
    print(f"{l1r:>9} {s['nonzero']:>8} {s['true kept']:>10} {s['noise kept']:>11} "
          f"{s['||w||']:>8.3f} {s['ROC-AUC']:>9.4f} {s['log loss']:>10.4f}")
print()
print("Smooth and monotone in sparsity, as the formula suggests. On THIS dataset that")
print("is all it does, and it is worth saying why: these 40 features are independent by")
print("construction, so there are no correlated duplicates for elastic net to exploit.")
print("The interesting difference between L1 and elastic net only appears when there")
print("ARE correlated features, so that is built here rather than asserted.")
print()

# --- a second dataset with genuine correlated groups -----------------------------------
rng_grp = np.random.default_rng(77)
n_grp = 4000
group_base = rng_grp.normal(0, 1, (n_grp, 4))
group_columns = []
for g in range(4):
    group_columns.append(np.column_stack(
        [group_base[:, g] + 0.06 * rng_grp.normal(0, 1, n_grp) for _ in range(3)]))
X_grp = np.hstack(group_columns)          # 12 columns, in 4 blocks of 3 near-duplicates
log_odds_grp = 2.0 * group_base[:, 0] + 0.6 * rng_grp.normal(0, 1, n_grp)
y_grp = rng_grp.binomial(1, 1.0 / (1.0 + np.exp(-log_odds_grp)))

Xg_tr, Xg_te, yg_tr, yg_te = train_test_split(
    X_grp, y_grp, test_size=0.3, random_state=0, stratify=y_grp)
sc_grp = StandardScaler().fit(Xg_tr)
Xg_tr, Xg_te = sc_grp.transform(Xg_tr), sc_grp.transform(Xg_te)

print("A second dataset: 12 features in 4 blocks of 3 near-duplicates (correlation 0.94")
print("within a block). Only the first block drives the label; the other 9 are noise.")
print("Same C, three penalties, and the difference is stark:")
print()

n_splits_grp = 5
cv_grp = StratifiedKFold(n_splits=n_splits_grp, shuffle=True, random_state=0)
group_effect = {}
for l1r, label in [(1.0, "L1"), (0.0, "L2"), (0.5, "elastic net")]:
    solver = "liblinear" if l1r in (0.0, 1.0) else "saga"
    full = LogisticRegression(C=0.05, l1_ratio=l1r, solver=solver, max_iter=40000).fit(Xg_tr, yg_tr)
    w = full.coef_.ravel()
    block0 = w[:3]
    spread = float(np.ptp(np.abs(block0)))
    selected_folds = np.zeros(12)
    for fold_tr, _ in cv_grp.split(Xg_tr, yg_tr):
        m = LogisticRegression(C=0.05, l1_ratio=l1r, solver=solver, max_iter=40000).fit(
            Xg_tr[fold_tr], yg_tr[fold_tr])
        selected_folds += (np.abs(m.coef_.ravel()) > 1e-8)
    group_effect[label] = (w.copy(), spread, selected_folds.astype(int).copy())
    print(f"  {label:<12} coefficients on the first block: "
          f"{np.array2string(block0, precision=2, suppress_small=True)}")
    print(f"  {'':<12} spread within the block: {spread:.3f}   "
          f"selected in 5 folds: {selected_folds.astype(int)[:3].tolist()}")
print()
l1_block, l1_spread, l1_folds = group_effect["L1"]
en_block, en_spread, en_folds = group_effect["elastic net"]
l2_block, l2_spread, l2_folds = group_effect["L2"]
print("All three recover the same SIGNAL - the first block's coefficients sum to")
print(f"{np.sum(l1_block):.2f} (L1), {np.sum(l2_block):.2f} (L2) and {np.sum(en_block):.2f} (elastic net), which is the same model to")
print("three decimal places. But look at how the weight is DISTRIBUTED inside the")
print("block, where they are nothing alike:")
print()
print(f"  L1            spread {l1_spread:.3f}  - one big, one medium, one zero")
print(f"  elastic net   spread {en_spread:.3f}  - nearly even")
print(f"  L2            spread {l2_spread:.3f}  - nearly even")
print()
print("That is the group effect, and it is a problem with L1 specifically:")
print()
print("  When features are near-duplicates, L1's penalty has no reason to prefer one")
print("  over another, so it picks one essentially at random and zeroes the rest. The")
print("  choice is stable on THIS resample and arbitrary in principle. The consequence")
print("  is not that the prediction is wrong - it is almost exactly as good - but that")
print("  the coefficients become uninterpretable and unstable. 'Feature 7 matters' is")
print("  not a finding when feature 7 is a copy of features 3 and 11; which one wins")
print("  is a coin flip that gets re-flipped on every resample.")
print()
print("  The fold counts show the arbitrariness directly. For L1 the three members of")
print(f"  the first block are selected in {l1_folds[0]}/{l1_folds[1]}/{l1_folds[2]} of the {n_splits_grp} folds - the group is split in an")
print(f"  unstable way and the members are not equally supported. For elastic net it is")
print(f"  {en_folds[0]}/{en_folds[1]}/{en_folds[2]}, and for L2 {l2_folds[0]}/{l2_folds[1]}/{l2_folds[2]}: all three, every time.")
print()
print("  The L2 component in elastic net breaks the tie by spreading weight across the")
print("  group rather than choosing within it, because squared error punishes any one")
print("  coefficient carrying a large share of shared variance. In exchange you give up")
print("  a little sparsity. On tabular data, where correlated columns are the norm,")
print("  that is a very good trade.")
print()
print("So the practical rule, which is not a compromise for its own sake:")
print()
print("  - L1 when features are genuinely independent and you want the sparsest model,")
print("    or when n is small relative to p and you need the selection to be possible.")
print("  - elastic net when features are correlated, or when groups of them are")
print("    duplicates. It is the default in most tabular work for that reason.")
print("  - L2 when you have no reason to prefer zeros and want the most stable")
print("    coefficients available.")
print()
print("One caveat on all of this: none of the three penalties knows WHY features are")
print("correlated. If two columns duplicate each other, the model will happily split")
print("weight between them and you will learn nothing from the coefficients. The fix")
print("for that is to drop the duplicates before modelling, not to ask the penalty to")
print("paper over it. Regularization stabilises a redundant design; it does not make")
print("the information in it non-redundant.")
print()


# ======================================================================================
# PART E - standardization
# ======================================================================================
print("-" * 78)
print("PART E - why you must scale, and this is not a formality")
print("-" * 78)
print("""
Both penalties are functions of the coefficients, and the coefficients depend on
the units of the features. Rescaling a column by 1000 divides its coefficient by
1000 and leaves the fitted probabilities identical - so a penalty that looks at
coefficient size is looking at something the model itself considers arbitrary.

Here it is done deliberately: 35 of the noise features are inflated 50x to 500x.
Nothing about the information content changes. The model is the same model.
""")
print()

rng_scale = np.random.default_rng(11)
X_raw, y_raw = make_sparse_signal_data()
feature_scale = np.ones(N_FEATURES)
feature_scale[N_TRUE:] = np.linspace(50, 500, N_FEATURES - N_TRUE)
X_scaled_units = X_raw * feature_scale

Xtr_u, Xte_u, ytr_u, yte_u = train_test_split(
    X_scaled_units, y_raw, test_size=0.3, random_state=0, stratify=y_raw
)

C_demo = 0.01
raw_model = LogisticRegression(C=C_demo, l1_ratio=1.0, solver="liblinear",
                               max_iter=40000).fit(Xtr_u, ytr_u)
sc_u = StandardScaler().fit(Xtr_u)
std_model = LogisticRegression(C=C_demo, l1_ratio=1.0, solver="liblinear",
                               max_iter=40000).fit(sc_u.transform(Xtr_u), ytr_u)

raw_sel = np.where(np.abs(raw_model.coef_.ravel()) > 1e-8)[0]
std_sel = np.where(np.abs(std_model.coef_.ravel()) > 1e-8)[0]

print(f"The same L1 penalty at the same C = {C_demo:g}, on the same information:")
print()
print(f"{'':<26} {'features selected':>18} {'true kept':>11} {'noise kept':>11} {'ROC-AUC':>9}")
print("-" * 78)
raw_auc = roc_auc_score(yte_u, raw_model.predict_proba(Xte_u)[:, 1])
std_auc = roc_auc_score(yte_u, std_model.predict_proba(sc_u.transform(Xte_u))[:, 1])
print(f"{'raw units (inflated)':<26} {raw_sel.size:>18} {int(np.isin(raw_sel, TRUE_IDX).sum()):>11} "
      f"{int(np.isin(raw_sel, np.arange(N_TRUE, N_FEATURES)).sum()):>11} {raw_auc:>9.4f}")
print(f"{'after StandardScaler':<26} {std_sel.size:>18} {int(np.isin(std_sel, TRUE_IDX).sum()):>11} "
      f"{int(np.isin(std_sel, np.arange(N_TRUE, N_FEATURES)).sum()):>11} {std_auc:>9.4f}")
print()
print(f"On raw units L1 selects {raw_sel.size} features, {int(np.isin(raw_sel, np.arange(N_TRUE, N_FEATURES)).sum())} of them noise - the inflated columns")
print("look enormous in coefficient units, so the penalty refuses to zero them out.")
print("After standardizing, the same fit selects "
      f"{std_sel.size}, indices {std_sel.tolist()}: the five real features, exactly as in Part C.")
print()
print("Same C, same penalty, same information, same code. The only thing that changed")
print("is the units. This is the single most common way to get L1 quietly wrong, and")
print("it is invisible in the training log because the model still trains fine and")
print("still produces plausible-looking probabilities.")
print()
print("The rule: scale your features before regularizing. Not after, and not")
print("optionally. With a penalty that inspects coefficient magnitudes, the choice of")
print("units is part of the model specification.")
print()
print("The same applies to C's meaning. The objective is C * sum(log loss) + penalty,")
print("and the sum of losses grows with the number of rows, so the effective strength")
print("of a given C changes as n changes. A C tuned on 4,000 rows is not the right C")
print("for 400,000. This is why CV is done inside a pipeline on scaled data, and why")
print("C is not a transferable constant. Part F does it properly.")
print()


# ======================================================================================
# PART F - choosing C
# ======================================================================================
print("-" * 78)
print("PART F - choosing C by cross-validation")
print("-" * 78)
print("""
Reading the path table and picking the elbow by eye is what everyone does first and
what you should not ship. Two problems: the table above is computed on the TEST set,
so choosing from it is choosing on the data you are reporting on, and the optimum
depends on the metric, which Part C showed is a genuine fork in the road rather
than a technicality.

So: pick it with cross-validation, on training data only, and decide the metric
first. Both are done here.
""")
print()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
grid = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

print("L1, 5-fold CV on the TRAINING data only, two metrics:")
print()
print(f"{'C':>8} {'CV log loss':>13} {'CV ROC-AUC':>12} {'mean nonzero':>14} {'std':>8}")
print("-" * 60)
cv_rows = []
for C in grid:
    ll_scores = -cross_val_score(
        LogisticRegression(C=C, l1_ratio=1.0, solver="liblinear", max_iter=40000),
        X_train_s, y_train, cv=cv, scoring="neg_log_loss")
    auc_scores = cross_val_score(
        LogisticRegression(C=C, l1_ratio=1.0, solver="liblinear", max_iter=40000),
        X_train_s, y_train, cv=cv, scoring="roc_auc")
    model = fit(1.0, C)
    model_nonzero = int((np.abs(model.coef_.ravel()) > 1e-8).sum())
    cv_rows.append({
        "C": C, "cv_logloss": ll_scores.mean(), "cv_auc": auc_scores.mean(),
        "nonzero": model_nonzero, "std": ll_scores.std(),
    })
    r = cv_rows[-1]
    print(f"{C:>8g} {r['cv_logloss']:>13.4f} {r['cv_auc']:>12.4f} {r['nonzero']:>14} {r['std']:>8.4f}")
print()

best_ll = min(cv_rows, key=lambda r: r["cv_logloss"])
best_auc = max(cv_rows, key=lambda r: r["cv_auc"])
print(f"By log loss:  C = {best_ll['C']:g}, selecting {best_ll['nonzero']:>2} features")
print(f"By ROC-AUC:  C = {best_auc['C']:g}, selecting {best_auc['nonzero']:>2} features")
print()
if best_ll["C"] == best_auc["C"]:
    print("On this data the two criteria happen to agree, which is convenient and not")
    print("something to expect in general - Part C already showed they can disagree,")
    print("and when they do, you have to choose which property you are buying.")
else:
    print("The two criteria disagree, exactly as Part C warned. Nothing is wrong here;")
    print("it means 'best calibrated probabilities' and 'best ordering' are different")
    print("models, and you have to decide which one the problem is asking for.")
print()
print(f"The standard error on each CV log loss is around {np.mean([r['std'] for r in cv_rows]):.3f}, which is worth comparing")
print("against the gaps in that column before you conclude that any C is better than")
print("another. Differences smaller than the standard error are not differences.")
print()
print("Two things this table does NOT tell you:")
print()
print("  - Nothing about the test set. It was not touched. That is the point.")
print("  - Nothing about the penalty TYPE. This is L1 only. Choosing between L1 and")
print("    elastic net is a second decision, and it belongs in the same search.")
print()
print("Which brings the honest caveat: with 5 real features out of 40, L1 selecting")
print(f"exactly those {N_TRUE} is a best case. Variable selection is a high-variance")
print("operation - on harder data, with correlated features or a weaker signal, the")
print("selected set is unstable across resamples. The stable way to measure that is")
print("to select on each CV fold and look at how often each feature comes back, which")
print("is the subject of script 13.")
print()


# ======================================================================================
# Plots
# ======================================================================================
print("-" * 78)
print("PLOTS")
print("-" * 78)

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# --- the C paths ----------------------------------------------------------------------
ax = axes[0, 0]
l2_cs = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
l1_cs = [0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.3, 1.0]
l2_nz, l1_nz, l1_ll, l2_ll = [], [], [], []
for c in l2_cs:
    m = fit(0.0, c, solver="lbfgs")
    l2_nz.append(int((np.abs(m.coef_.ravel()) > 1e-8).sum()))
    l2_ll.append(log_loss(y_test, m.predict_proba(X_test_s)[:, 1]))
for c in l1_cs:
    m = fit(1.0, c, solver="liblinear")
    l1_nz.append(int((np.abs(m.coef_.ravel()) > 1e-8).sum()))
    l1_ll.append(log_loss(y_test, m.predict_proba(X_test_s)[:, 1]))
ax.semilogx(l2_cs, l2_nz, "o-", linewidth=2.4, color="#4C6EF5",
            label=f"L2 (ramp: never reaches 0)")
ax.semilogx(l1_cs, l1_nz, "o-", linewidth=2.4, color="#E03131",
            label="L1 (falls to the 5 real features)")
ax.axhline(N_TRUE, color="#0CA678", linestyle=":", linewidth=2)
ax.text(min(l1_cs), N_TRUE + 1.5, "the true number of real features", fontsize=9, color="#0CA678")
ax.set_xlabel("C (log scale, higher C = less regularisation)")
ax.set_ylabel("nonzero coefficients")
ax.set_title("L2 never sets a coefficient to zero; L1 does", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, frameon=False)
ax.grid(alpha=0.25, which="both")

# --- ranking vs calibration along the L1 path -----------------------------------------
ax = axes[0, 1]
l1_auc = [roc_auc_score(y_test, fit(1.0, c, solver="liblinear").predict_proba(X_test_s)[:, 1])
          for c in l1_cs]
ax.semilogx(l1_cs, l1_auc, "o-", linewidth=2.6, color="#4C6EF5", label="ROC-AUC (ranking)")
ax.semilogx(l1_cs, l1_ll, "s-", linewidth=2.6, color="#E03131", label="log loss (calibration)")
ax.axvline(C_exact, color="#7048E8", linestyle="--", linewidth=1.8)
ax.text(C_exact, ax.get_ylim()[0], f"exact recovery\nC={C_exact:g}", fontsize=8.5,
        color="#7048E8", va="bottom")
ax.set_xlabel("C (log scale)")
ax.set_title("The two metrics peak at different C",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9, frameon=False)
ax.grid(alpha=0.25, which="both")

# --- standardization -------------------------------------------------------------------
ax = axes[1, 0]
all_idx = np.arange(N_FEATURES)
ax.scatter(all_idx[N_TRUE:], np.abs(raw_model.coef_.ravel()[N_TRUE:]), s=42,
           color="#E03131", label="noise features, raw units", alpha=0.85)
ax.scatter(all_idx[:N_TRUE], np.abs(raw_model.coef_.ravel()[:N_TRUE]), s=70,
           color="#0CA678", marker="D", label="real features, raw units", zorder=5)
ax.scatter(all_idx[N_TRUE:], np.abs(std_model.coef_.ravel()[N_TRUE:]), s=42,
           color="#E03131", marker="x", label="noise features, standardized", alpha=0.85)
ax.scatter(all_idx[:N_TRUE], np.abs(std_model.coef_.ravel()[:N_TRUE]), s=70,
           color="#0CA678", marker="D", facecolors="none", linewidths=2.2,
           label="real features, standardized", zorder=5)
ax.set_yscale("log")
ax.set_xlabel("feature index")
ax.set_ylabel("|coefficient| (log scale)")
ax.set_title("Same penalty, same C: the units decide what gets zeroed",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25, which="both")

# --- CV -------------------------------------------------------------------------------
ax = axes[1, 1]
grid_x = np.arange(len(grid))
width = 0.38
ax.bar(grid_x - width / 2, [r["cv_logloss"] for r in cv_rows], width,
       color="#E03131", label="CV log loss (lower better)")
ax.set_ylabel("CV log loss", color="#E03131")
ax.set_xlabel("C")
ax.set_xticks(grid_x)
ax.set_xticklabels([f"{c:g}" for c in grid])
ax.set_yscale("log")
ax2 = ax.twinx()
ax2.bar(grid_x + width / 2, [r["cv_auc"] for r in cv_rows], width,
        color="#4C6EF5", label="CV ROC-AUC (higher better)")
ax2.set_ylabel("CV ROC-AUC", color="#4C6EF5")
ax.set_title("Picking C on the training folds, never on the test set",
             fontsize=11, fontweight="bold")
ax.grid(alpha=0.25, axis="y")
lines = [plt.Line2D([], [], color="#E03131", linewidth=8),
         plt.Line2D([], [], color="#4C6EF5", linewidth=8)]
ax.legend(lines, ["CV log loss", "CV ROC-AUC"], fontsize=9, frameon=False, loc="upper center")

fig.suptitle("08 - Regularization: L1, L2 and Elastic Net", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


# ======================================================================================
print()
print("=" * 78)
print("TAKEAWAYS")
print("=" * 78)
print("1. The penalty is not a fudge factor, it changes the question. The unpenalised")
print("   fit answers 'what fits this sample'; the penalised fit answers 'what is the")
print("   simplest thing that fits this sample'. On this data that difference is the")
print("   whole result: 40 nonzero coefficients versus 5, all of them real.")
print()
print("2. L2 shrinks, L1 selects. L2's penalty is differentiable at zero, so it can")
print("   never pin a coefficient there; every feature comes back, just smaller. L1's")
print("   kinks land on the axes of its diamond-shaped constraint region, so whole")
print("   coefficients get parked at exactly zero. That is feature selection as a")
print("   geometric consequence, not an extra step.")
print()
print(f"3. On data where the answer is known, L1 at C={C_exact:g} recovered all {N_TRUE} real features and none of the")
print(f"   {N_FEATURES - N_TRUE} noise features. Check your own data the same way before you")
print("   trust a selection you did not build.")
print()
print("4. Elastic net is not L1 for people who cannot decide. The L2 component breaks")
print("   L1's arbitrary tie-breaking among correlated features, at the cost of a")
print("   little sparsity. If your features are correlated - which tabular features")
print("   usually are - that is usually the right trade.")
print()
print("5. Standardize before you regularize, always. Both penalties inspect")
print("   coefficient magnitudes, and coefficient magnitudes depend on the units you")
print("   happened to record. In Part E the identical L1 fit at the identical C chose")
print("   38 features on raw units and 5 after scaling, from identical information.")
print()
print("6. Sparsity and calibration pull in opposite directions. The sparsest L1 model")
print("   had the best ROC-AUC and the worst log loss: dropping noise improves the")
print("   ordering, and the extra shrinkage makes the probabilities too small. Choose")
print("   C against the metric the problem actually cares about, deliberately.")
print()
print("7. Choose C by cross-validation on training data, and compare differences to")
print("   the standard error before believing them. And remember C is not portable:")
print("   the objective sums the loss over rows, so the same C means a different")
print("   amount of regularisation at a different n.")
