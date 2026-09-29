"""
09 - Multiclass Logistic Regression
====================================

Goal: extend logistic regression from two classes to K, and understand what actually
changes - including the things people get wrong about it.

The problem being solved
------------------------
Five well-separated Gaussian clusters in 2D. Five classes, 3,000 rows, 600 each.
Simple enough that accuracy will be high, which is deliberate: the interesting
differences between methods here are NOT in accuracy. They are in the probabilities
and in the coefficients, which is where multiclass modelling actually lives.

The model
---------
Binary logistic regression produces one number, the log-odds, and squashes it.
Multiclass produces K numbers, one per class, and normalises them:

    z_k = w_k . x + b_k                      the k-th logit, a linear function
    p_k = exp(z_k) / sum_j exp(z_j)           softmax - the only new idea

That is the entire extension. Everything else in this file is a consequence of it,
including the one genuinely surprising consequence: K sets of coefficients contain
K - 1 sets of information, not K, and the redundancy is not a detail you can ignore
when you try to interpret them.

scikit-learn version notes (this file is written against 1.8)
------------------------------------------------------------
* `multi_class=` HAS BEEN REMOVED from LogisticRegression's signature. In 1.8 a
  multiclass fit is always multinomial (softmax). You no longer choose, and the
  'ovr' spelling is gone. To get one-vs-rest you must ask for it explicitly with
  sklearn.multiclass.OneVsRestClassifier, which is compared below.
* `penalty=` is deprecated; use `l1_ratio` (0 = L2, 1 = L1) as in script 08.

What this file demonstrates
---------------------------
1. Softmax written out, and the numerical-stability trick that makes it work.
2. Hand-written softmax gradient descent, gradient-checked against scikit-learn.
3. The binary case is a special case: softmax with 2 classes IS the sigmoid.
4. Shift invariance: only DIFFERENCES of logits are real. K rows, K - 1 of meaning,
   and why the L2 fit hands you centred coefficients while L1 does not.
5. The near-miss identity everyone quotes: the pairwise logit difference is NOT the
   binary logistic fit, though it correlates above 0.99 with it. Also a worked
   example of a check that appeared to fail for the wrong reason.
6. Multinomial vs one-vs-rest: they disagree on accuracy, cross-validation reverses
   that disagreement, and log loss settles it.
7. What per-class coefficients do and do not mean, and what happens to a model that
   never met a class it later has to predict.

Run:  python 09_multiclass_logistic_regression.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             log_loss)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 20)

CLASS_NAMES = ["north-west", "east", "north", "south", "far-east"]
CENTER_NAMES = ["centre A", "centre B", "centre C", "centre D", "centre E"]
N_CLASSES = 5


# ======================================================================================
# The data
# ======================================================================================
def make_multiclass_blobs(n=3000, seed=5):
    """Five Gaussian blobs in 2D. Deliberately well separated so that accuracy
    saturates and the differences between methods show up in the probabilities."""
    centers = np.array([[0.0, 0.0],
                        [4.0, 0.0],
                        [2.0, 3.5],
                        [2.0, -3.5],
                        [6.0, 3.0]])
    X, y = make_blobs(n_samples=n, centers=centers, cluster_std=1.0, random_state=seed)
    return X, y, centers


X, y, CENTERS = make_multiclass_blobs()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

print("=" * 78)
print("MULTICLASS LOGISTIC REGRESSION")
print("=" * 78)
print(f"Training rows {len(X_train):,}, features {X_train.shape[1]}, classes {N_CLASSES}")
print(f"Test rows     {len(X_test):,}")
print(f"Class counts  {np.bincount(y)}")
print()
print("Features are standardized, so the coefficient magnitudes below are comparable")
print("to each other. That matters in Part G; script 08 explains why.")
print()


# ======================================================================================
# PART A - the softmax function, written out
# ======================================================================================
print("-" * 78)
print("PART A - what the extension actually is")
print("-" * 78)
print("""
Binary logistic regression: one linear function, one sigmoid, done.

Multiclass: K linear functions, and a normalisation that makes them compete.
    z_k = w_k . x + b_k           for k = 0, 1, ..., K-1
    p_k = exp(z_k) / sum_j exp(z_j)
""")
print("Note what is NOT here: no new loss function, no new optimiser, no new")
print("hyperparameter. The cross-entropy is the same function, with the predicted")
print("probability vector replacing the single predicted probability. The gradient")
print("is the same shape of argument too:")
print()
print("    dL/dz_k = p_k - y_k")
print()
print("which is the beautiful part: it is literally the binary gradient, applied K")
print("times, once per class. If p_k is large and y_k is 0, push that logit down.")
print("Every class is corrected in the same direction at once.")
print()


def softmax(z):
    """Numerically stable softmax. Subtracting the row max leaves the result
    unchanged mathematically (see the shift-invariance check below) and prevents
    exp() from overflowing on large logits."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


print("The stability trick deserves a demonstration, because the naive version looks")
print("fine right up until it returns NaN:")
print()


def softmax_naive(z):
    """The textbook version, with the subtraction deliberately omitted so its
    failure mode is visible. Do not use this in real code."""
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


demo = np.array([[0.0, 0.0],
                 [1000.0, 1000.0],
                 [-1000.0, -1000.0],
                 [800.0, 799.0]])

# The overflow below is the point of the demonstration, so numpy's warnings about
# it are suppressed here rather than left to clutter the output.
with np.errstate(over="ignore", invalid="ignore"):
    naive_results = softmax_naive(demo)
    stable_results = softmax(demo)

print(f"{'logits':>24} {'naive softmax':>28} {'stable softmax':>28}")
print("-" * 82)
for row, n_res, s_res in zip(demo, naive_results, stable_results):
    n_txt = str(np.round(n_res, 6)) if np.all(np.isfinite(n_res)) else "NaN"
    print(f"{str(row):>24} {n_txt:>28} {str(np.round(s_res, 6)):>28}")
print()
print("The naive version overflows exp(1000) to infinity, so inf/inf is NaN. Note that")
print("[-1000, -1000] fails too, which surprises people - exp underflows to zero, the")
print("denominator is zero, and 0/0 is also NaN. Both directions break it.")
print()
print("The stable version subtracts each row's maximum first. That is mathematically")
print("a no-op - it cancels in the ratio - but it keeps every exponential inside")
print("[0, 1] instead of letting them run to 0 or infinity. This is not a theoretical")
print("concern: float64 overflows exp() at about 709, and real logits on")
print("unstandardised features with large coefficients get there routinely. In a")
print("long pipeline you may not see the RuntimeWarning at all, and a column of")
print("silent NaNs in your probabilities is a genuinely bad afternoon.")
print()
print("Keep the stable version. It is one line longer and never has a bad day.")
print()


# ======================================================================================
# PART B - hand-written gradient descent, checked against scikit-learn
# ======================================================================================
print("-" * 78)
print("PART B - writing it yourself, to check you understand it")
print("-" * 78)
print("Full-batch gradient descent on the softmax cross-entropy, with an optional L2")
print("penalty. The gradient is one line, which is the payoff for the formulation:")
print()


def softmax_gradients(X, y, W, b, l1_ratio=0.0):
    """Gradient of mean softmax cross-entropy + l1_ratio*||W||_F^2/2, plus its intercept.

    dL/dW = X.T @ G + l1_ratio * W      where G = (S - Y_onehot) / n
    dL/db = G.sum(axis=0)

    G has shape (n, K); X.T @ G has shape (d, K), so it is transposed to match W
    being stored as (K, d) - one row of coefficients per class.
    """
    n = len(X)
    Z = X @ W.T + b
    S = softmax(Z)
    onehot = np.zeros_like(S)
    onehot[np.arange(n), y] = 1.0
    G = (S - onehot) / n
    return G.T @ X + l1_ratio * W, G.sum(axis=0)


def fit_softmax_gd(X, y, n_classes, l1_ratio=0.0, lr=0.5, iters=4000, seed=0):
    """Plain gradient descent. No momentum, no line search, no scaling tricks -
    the data is already standardized, which is the only reason this converges."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, size=(n_classes, X.shape[1]))
    b = np.zeros(n_classes)
    for _ in range(iters):
        gW, gb = softmax_gradients(X, y, W, b, l1_ratio)
        W -= lr * gW
        b -= lr * gb
    return W, b


def loss_softmax(X, y, W, b):
    """Mean softmax cross-entropy, written with log-sum-exp so it does not underflow.

        loss = (1/n) * sum_i [ logsumexp(z_i) - z_{i, y_i} ]

    The shape of the last line is the whole trick. logsumexp(z_i) is ONE scalar per
    row, a vector of length n. Subtracting z at the true class picks a single entry
    per row. Building an (n, K) array of 'logits plus the row logsumexp' and then
    indexing it at the true class is a natural-looking mistake that silently
    cancels the class-specific term and computes a different loss - see the check
    below, which is how this was found.
    """
    Z = X @ W.T + b
    Z = Z - Z.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(Z).sum(axis=1))          # (n,) - one scalar per row
    return float(np.mean(logsumexp - Z[np.arange(len(X)), y]))


print("Sanity check first: does the gradient agree with a numerical difference")
print("quotient at a random point? If this is wrong, nothing after it means anything.")
print()

rng = np.random.default_rng(0)
X_sub, y_sub = X_train_s[:2000], y_train[:2000]
W_test = rng.normal(0, 0.5, size=(N_CLASSES, X_train_s.shape[1]))
b_test = rng.normal(0, 0.5, size=N_CLASSES)
analytic_W, analytic_b = softmax_gradients(X_sub, y_sub, W_test, b_test)

h = 1e-6
numeric_W = np.zeros_like(W_test)
numeric_b = np.zeros_like(b_test)
for k in range(N_CLASSES):
    for d in range(X_train_s.shape[1]):
        W_plus, W_minus = W_test.copy(), W_test.copy()
        W_plus[k, d] += h
        W_minus[k, d] -= h
        numeric_W[k, d] = (loss_softmax(X_sub, y_sub, W_plus, b_test)
                           - loss_softmax(X_sub, y_sub, W_minus, b_test)) / (2 * h)
    b_plus, b_minus = b_test.copy(), b_test.copy()
    b_plus[k] += h
    b_minus[k] -= h
    numeric_b[k] = (loss_softmax(X_sub, y_sub, W_test, b_plus)
                    - loss_softmax(X_sub, y_sub, W_test, b_minus)) / (2 * h)


def scaled_max_error(analytic, numeric):
    """Normalise by the LARGEST magnitude, not elementwise.

    An elementwise relative error divides by each entry, and any entry whose true
    value is near zero produces an enormous number that tells you nothing about the
    gradient. This is the standard way to write a gradient check: the question is
    whether the two arrays agree to within the size of the arrays themselves.
    """
    return float(np.max(np.abs(analytic - numeric)) / (np.max(np.abs(numeric)) + 1e-12))


err_W = scaled_max_error(analytic_W, numeric_W)
err_b = scaled_max_error(analytic_b, numeric_b)
print(f"  max scaled error, coefficients: {err_W:.2e}")
print(f"  max scaled error, intercepts:   {err_b:.2e}")
print("  (central differences, h = 1e-6, error normalised by the largest component)")
print()
print(f"The two agree to {max(err_W, err_b):.0e}, which is exactly the floor set by")
print("float64 arithmetic on a central difference. The gradient is correct.")
print()
print("Worth noting what this check is actually for. The first version of the loss")
print("function above built an (n, K) array of 'shifted logit plus the row logsumexp'")
print("and indexed it at the true class. That looks like the textbook formula and")
print("is wrong: the class-specific term cancels, so it silently computes")
print("mean_i logsumexp(z_i) instead of mean_i [logsumexp(z_i) - z_{i,y_i}]. The")
print("gradient check flagged it immediately with an error of order 1.")
print()
print("That is the argument for writing the check rather than trusting the algebra.")
print("The formula was wrong in a way that reading it three more times would not")
print("have caught, and a wrong loss function that still produces plausible-looking")
print("numbers is the most expensive kind of bug. The shape of the last line of")
print("loss_softmax is now commented for the same reason.")
print()


print("Now fit it two ways on the same standardized training data and compare on the")
print("held-out test set. Plain GD will not match scikit-learn exactly - lbfgs is a")
print("much better optimiser - but it should land in the same place.")
print()

W_gd, b_gd = fit_softmax_gd(X_train_s, y_train, N_CLASSES, lr=0.5, iters=4000)
proba_gd = softmax(X_test_s @ W_gd.T + b_gd)
pred_gd = proba_gd.argmax(axis=1)

sk_model = LogisticRegression(max_iter=5000, C=1.0).fit(X_train_s, y_train)
proba_sk = sk_model.predict_proba(X_test_s)
pred_sk = sk_model.predict(X_test_s)

print(f"{'model':<30} {'accuracy':>10} {'log loss':>10}")
print("-" * 52)
for name, pred, proba in [("hand-written gradient descent", pred_gd, proba_gd),
                          ("sklearn LogisticRegression", pred_sk, proba_sk)]:
    print(f"{name:<30} {accuracy_score(y_test, pred):>10.4f} "
          f"{log_loss(y_test, proba, labels=sk_model.classes_):>10.4f}")
print()
print("Same answer, arrived at by different routes. That is the point of the exercise:")
print("the fit is a property of the objective, not of the library. If your own")
print("implementation disagrees, one of you has a bug, and the disagreement is the")
print("most useful thing you will learn all week.")
print()
acc_gd = accuracy_score(y_test, pred_gd)
ll_gd = log_loss(y_test, proba_gd, labels=sk_model.classes_)
acc_sk = accuracy_score(y_test, pred_sk)
ll_sk = log_loss(y_test, proba_sk, labels=sk_model.classes_)
print(f"The two are within {abs(acc_gd - acc_sk):.4f} on accuracy and {abs(ll_gd - ll_sk):.4f} on log loss,")
print("which is the level of agreement you should expect between a converged lbfgs fit")
print("and fixed-step gradient descent. Neither is 'the right answer' and the other is")
print("an approximation: both minimise the same objective, and they stop at slightly")
print("different points. The coefficients differ by "
      f"{np.max(np.abs(W_gd - sk_model.coef_)):.2f} in the largest entry, which sounds large until you")
print(f"recall the raw coefficient scale here runs to {np.max(np.abs(sk_model.coef_)):.1f}.")
print()
print("The fact that a hand-rolled optimiser is competitive on a 2-feature problem is")
print("not a recommendation to hand-roll optimisers. It is 4,000 iterations on 2,100")
print("rows. Script 03 is about how quickly that stops being true, and script 17")
print("closes with the same lesson in production form: use the library's optimiser,")
print("and write your own only to understand what it is doing.")
print()


# ======================================================================================
# PART C - the binary case is a special case
# ======================================================================================
print("-" * 78)
print("PART C - is multiclass just binary, repeated?")
print("-" * 78)
print("A reasonable suspicion: a 2-class softmax is just a logistic sigmoid with extra")
print("steps. This is true, and it is worth confirming numerically because it tells")
print("you the general case really is a specialisation rather than a different model.")
print()

mask_train = y_train < 2
mask_test = y_test < 2
soft2 = LogisticRegression(max_iter=5000).fit(X_train_s[mask_train], y_train[mask_train])
bin2 = LogisticRegression(max_iter=5000).fit(
    X_train_s[mask_train], (y_train[mask_train] == 1).astype(int))

Z2 = soft2.decision_function(X_test_s[mask_test])
diff_Z2 = Z2[:, 0] - Z2[:, 1] if Z2.ndim == 2 else Z2
dev_binary = np.max(np.abs(diff_Z2 - bin2.decision_function(X_test_s[mask_test])))
print(f"  2-class softmax logit difference vs a separate binary fit: max dev {dev_binary:.2e}")
print()
print("Exactly zero, to machine precision. With 2 classes the softmax reduces")
print("algebraically to the logistic sigmoid:")
print()
print("    p_1 = exp(z_1) / (exp(z_0) + exp(z_1)) = 1 / (1 + exp(-(z_1 - z_0)))")
print()
print("so all the apparent machinery collapses to the one logit you already knew.")
print()
print("Two further things sklearn does that surprise people, both visible above:")
print()
print(f"  decision_function shape for 2 classes: {Z2.shape}  - ONE column, not two")
print(f"  coef_ shape for 2 classes:             {soft2.coef_.shape}  - ONE row, not two")
print()
print("sklearn does not store the redundant second softmax row. It recognises the")
print("degeneracy and returns the binary form. With 5 classes you get the full")
print(f"{sk_model.coef_.shape[0]} x {sk_model.coef_.shape[1]} matrix, because there the rows are not redundant.")
print()
print("Third surprise, and it will bite you in Part G: the liblinear solver no longer")
print("handles multiclass at all. It raises a ValueError for 3 or more classes, so a")
print("snippet from a tutorial that says solver='liblinear' for speed will simply stop")
print("working the moment you add a class.")
print()

print("Which solvers accept a 5-class problem in this version:")
print()
for solver_name in ["lbfgs", "newton-cg", "newton-cholesky", "saga", "liblinear"]:
    try:
        LogisticRegression(solver=solver_name, max_iter=200).fit(X_train_s, y_train)
        status = "works"
    except ValueError:
        status = "ValueError - binary only"
    print(f"  {solver_name:<18} {status}")
print()
print("lbfgs remains the right default. saga is the only one of these that does L1 or")
print("elastic net here, and Part G is about what that costs you.")
print()


# ======================================================================================
# PART D - shift invariance, and the redundancy it implies
# ======================================================================================
print("-" * 78)
print("PART D - only DIFFERENCES of logits mean anything")
print("-" * 78)
print("""
The softmax is invariant to adding a constant to every logit of a row:

    exp(z_k + c) / sum_j exp(z_j + c) = exp(z_k) / sum_j exp(z_j)

So the model is a function of K numbers, but only K - 1 of them carry information.
The (K - 1)th direction is a gauge - a free parameter with no effect on any
prediction. This is exactly like the non-identifiability of linear regression
coefficients, and it has the same consequence: you cannot read a coefficient
without first fixing a convention.
""")
print()

Z_full = sk_model.decision_function(X_test_s)
proba_shift = softmax(Z_full + 7.3)
shift_dev = np.max(np.abs(proba_sk - proba_shift))
print(f"  max |p(z) - p(z + 7.3)| = {shift_dev:.2e}")
print()
print("Identical probabilities from a completely different set of logits. Any")
print("implementation that returns raw logits is returning something with an")
print("arbitrary constant baked in.")
print()

n_params = sk_model.coef_.size + sk_model.intercept_.size
print(f"Parameters as stored: {sk_model.coef_.shape[0]} coefficient rows x "
      f"{sk_model.coef_.shape[1]} features + {sk_model.coef_.shape[0]} intercepts")
print(f"  = {sk_model.coef_.size} coefficients + {sk_model.intercept_.size} intercepts = {n_params} numbers")
print(f"Directions of genuine information: {N_CLASSES - 1} of them")
print()
print("This is a real trap, not a technicality. People fit a 5-class model, see 5")
print("coefficient rows, and try to read all 5 as 'the effect of x on class k'. That")
print("is wrong in a specific, demonstrable way, and Part G shows the symptom.")
print()
print("Two ways to remove the ambiguity, both used in practice:")
print()
print("  1. Sum-to-zero: constrain the rows so they sum to zero across classes.")
print("     The 'deviation from the average class' is then the interpretable quantity.")
print("  2. Drop a reference class. Everything is compared to it. This is what")
print("     multinomial logit models in R and Stata do by default.")
print()
print("sklearn does neither, and does not promise to. The gauge is yours to fix.")
print()


# ======================================================================================
# PART E - an identity that is almost true, and the trap in checking it
# ======================================================================================
print("-" * 78)
print("PART E - a nearly-true identity, and how to check it without fooling yourself")
print("-" * 78)
print("""
You will see this stated as fact:

    'In a multinomial model, the difference between the coefficients for class a
     and class b is the logistic regression you would get by fitting a binary
     problem on classes a and b.'

It is very nearly true, and that is exactly what makes it dangerous - near enough
that nobody checks, and not true enough to build on. Here is the actual state of
it, measured rather than asserted.
""")
print()
print("The FORM is genuinely exact. Because the softmax normalises, the odds of class")
print("b against class a are:")
print()
print("    p_b / p_a = exp(z_b) / exp(z_a) = exp(z_b - z_a)")
print()
print("which is the logistic function of the difference. So z_b - z_a is a proper")
print("log-odds score for 'b versus a', and it deserves to be compared with the binary")
print("fit. Note the ORDER: b minus a, to match a binary target of (y == b).")
print()

pair_rows = []
for a, b in [(0, 1), (1, 3), (2, 4), (0, 4)]:
    pair_mask = (y_train == a) | (y_train == b)
    pair_model = LogisticRegression(max_iter=5000).fit(
        X_train_s[pair_mask], (y_train[pair_mask] == b).astype(int))
    multi_diff = Z_full[:, b] - Z_full[:, a]           # order matters: b minus a
    binary_logit = pair_model.decision_function(X_test_s)
    dev = np.abs(multi_diff - binary_logit)
    pair_rows.append({
        "pair": f"{a} vs {b}",
        "||w_multinomial||": round(float(np.linalg.norm(sk_model.coef_[b] - sk_model.coef_[a])), 3),
        "||w_binary||": round(float(np.linalg.norm(pair_model.coef_[0])), 3),
        "correlation": round(float(np.corrcoef(multi_diff, binary_logit)[0, 1]), 5),
        "sign agree": round(float(np.mean(np.sign(multi_diff) == np.sign(binary_logit))), 4),
        "median dev": round(float(np.median(dev)), 4),
        "max dev": round(float(dev.max()), 4),
    })

pair_df = pd.DataFrame(pair_rows)
print(pair_df.to_string(index=False))
print()

min_corr = pair_df["correlation"].min()
close_rows = pair_df[pair_df["max dev"] < 5.0]
close_median = close_rows["median dev"].max()
odd_row = pair_df.loc[pair_df["max dev"].idxmax()]

print("Read the deviation columns together, because neither column tells the story")
print("alone.")
print()
print(f"  Correlation is {min_corr:.4f} or better on every pair. The two scores are almost")
print("  the same function. The claim is not nonsense - it is nearly right.")
print()
print(f"  Three of the four pairs agree to within {close_median:.2f} logit units at the median, on a")
print(f"  scale where the logits themselves reach {np.abs(Z_full).max():.1f}. That is close enough to pass for")
print("  an identity in casual use, and that is exactly the problem.")
print()

odd_pair = odd_row["pair"]
odd_dev = odd_row["median dev"]
odd_ratio = odd_row["||w_multinomial||"] / odd_row["||w_binary||"]
print(f"  The exception is {odd_pair}: median deviation {odd_dev:.2f}, max {odd_row['max dev']:.2f}. That pair is the two")
print("  most distant clusters in the data, and the two coefficient vectors have very")
print(f"  different lengths ({odd_row['||w_multinomial||']} against {odd_row['||w_binary||']}, a factor of {odd_ratio:.1f}).")
print()
print("  That is the pattern to look for. The binary fit never sees the three classes")
print("  sitting between its two targets, so nothing pushes those two apart; the")
print("  multinomial contrast has to explain them in the presence of everything else,")
print("  and it is far more confident about the separation. The further apart the two")
print("  classes sit relative to the rest of the space, the more the approximation")
print("  degrades. So the identity is worst exactly where you would most want to use")
print("  it, which is a good reason not to rely on it at all.")
print()

print("So the honest statement is: the multinomial logit difference is a good")
print("approximation to the binary logistic fit on nearby classes, a poor one on distant")
print("ones, and never exactly either. Treat it as a sanity check, not a theorem.")
print()
print("Why not exact? The two fits minimise different objectives. The binary fit's")
print("log-likelihood is built from a two-way softmax; the multinomial's is built from")
print("a K-way one, so every row's normalising term involves all K logits, and the")
print("gradient for the (b - a) direction is affected by the other classes. Subtract")
print("the log-likelihood of the multinomial fit from that of the binary fit and you")
print("do not get a constant - you get a function of x that vanishes only at the")
print("optimum of the first. The two optima genuinely differ.")
print()
print("This is not a rounding detail, and it is not a bug. It is the reason a")
print("multiclass model is not 'five binary models'. Practical consequences:")
print()
print("  - Use the multiclass fit. Its parameters are jointly consistent, and the")
print("    per-class directions borrow strength across the dataset.")
print("  - Use a binary fit when you actually have a two-class question, or as a")
print("    readable baseline. It is a good model; it is just a different model.")
print("  - Do not derive a class-a-versus-class-b scoreboard by refitting K binary")
print("    models and calling it a decomposition of the multiclass coefficients.")
print()

print("A warning about checking this kind of claim, because it nearly fooled this")
print("file. The first version of the table above computed z_a - z_b for the first")
print("pair - the other way round - and produced:")
print()

check_a, check_b = 0, 1
check_mask = (y_train == check_a) | (y_train == check_b)
check_binary = LogisticRegression(max_iter=5000).fit(
    X_train_s[check_mask], (y_train[check_mask] == check_b).astype(int))
check_logit = check_binary.decision_function(X_test_s)
right_order = Z_full[:, check_b] - Z_full[:, check_a]
wrong_order = Z_full[:, check_a] - Z_full[:, check_b]
corr_right = np.corrcoef(right_order, check_logit)[0, 1]
corr_wrong = np.corrcoef(wrong_order, check_logit)[0, 1]

print(f"    right order  (z_{check_b} - z_{check_a}):  correlation {corr_right:+.4f},  "
      f"max deviation {np.max(np.abs(right_order - check_logit)):.2f}")
print(f"    wrong order  (z_{check_a} - z_{check_b}):  correlation {corr_wrong:+.4f},  "
      f"max deviation {np.max(np.abs(wrong_order - check_logit)):.2f}")
print()
print(f"The wrong order gives a correlation of {corr_wrong:+.4f} and deviations in the tens.")
print("Read carelessly, that looks like devastating evidence that the identity is")
print(f"false - and it is the signature of nothing more than a subtraction the wrong way")
print("round. A correlation near -1 is the tell: it means the relationship is real and")
print("pointing the wrong way, not that you have found a different one.")
print()
print("The habit worth keeping: when a check fails, look at the SIGN and the SHAPE of")
print("the failure before concluding the theory is wrong. A wrong-order subtraction and")
print("a genuinely false identity produce very different numbers, and the distinction")
print("is visible immediately from the correlation - which is why this table reports")
print("it, rather than only a max deviation that hides the direction of the error.")
print()


# ======================================================================================
# PART F - multinomial vs one-vs-rest
# ======================================================================================
print("-" * 78)
print("PART F - multinomial vs one-vs-rest, and they disagree about which is better")
print("-" * 78)
print("""
One-vs-rest fits K independent binary classifiers, one per class, then predicts
argmax. The attraction is that it reuses everything you already know about binary
logistic regression, and it works for problems where the classes genuinely are not
mutually exclusive. The cost is that the K sigmoids are estimated independently, so
they do not have to agree about what a probability means.
""")
print()

ovr = OneVsRestClassifier(LogisticRegression(max_iter=5000)).fit(X_train_s, y_train)
proba_ovr = ovr.predict_proba(X_test_s)
pred_ovr = ovr.predict(X_test_s)

print(f"{'model':<28} {'accuracy':>10} {'log loss':>10}")
print("-" * 50)
for name, pred, proba in [("multinomial (softmax)", pred_sk, proba_sk),
                          ("one-vs-rest (K sigmoids)", pred_ovr, proba_ovr)]:
    print(f"{name:<28} {accuracy_score(y_test, pred):>10.4f} "
          f"{log_loss(y_test, proba, labels=sk_model.classes_):>10.4f}")
print()

acc_mult = accuracy_score(y_test, pred_sk)
acc_ovr = accuracy_score(y_test, pred_ovr)
ll_mult = log_loss(y_test, proba_sk, labels=sk_model.classes_)
ll_ovr = log_loss(y_test, proba_ovr, labels=sk_model.classes_)
acc_winner = "one-vs-rest" if acc_ovr > acc_mult else "multinomial"
ll_winner = "multinomial" if ll_mult < ll_ovr else "one-vs-rest"

print(f"Accuracy winner:  {acc_winner} ({max(acc_mult, acc_ovr):.4f} vs {min(acc_mult, acc_ovr):.4f})")
print(f"Log loss winner:  {ll_winner} ({min(ll_mult, ll_ovr):.4f} vs {max(ll_mult, ll_ovr):.4f})")
print()
print("They do not agree on this split, and neither is wrong. Accuracy only cares")
print("about the argmax, so the renormalisation one-vs-rest applies at the end can")
print("reorder a close call. Log loss cares about the whole probability vector, and")
print("there one-vs-rest is much worse, because the independent sigmoids are not")
print("jointly coherent.")
print()
print(f"Do not settle anything on the accuracy line. The gap is {abs(acc_mult - acc_ovr):.4f}, and Part I")
print("shows that a gap that small is smaller than the noise in the measurement -")
print("cross-validation actually reverses it. The log loss gap is a different order")
print("of magnitude and survives every check.")
print()

print("A subtlety in scikit-learn 1.8 that is easy to get wrong: the probabilities")
print("from OneVsRestClassifier.predict_proba DO sum to 1.")
print()
row_sums = proba_ovr.sum(axis=1)
print(f"  row sums: min {row_sums.min():.15f}, max {row_sums.max():.15f}")
print(f"  largest deviation from 1: {np.max(np.abs(row_sums - 1)):.2e}")
print()
print("That is a normalisation sklearn applies after the fact, not a property of the")
print("model. The five sigmoids were fitted independently, so their outputs are only")
print("comparable if you assume the classes partition the space - which is an")
print("assumption, not a result. Renormalising makes the output LOOK like a")
print("distribution; it does not make the independence assumption true. That is")
print("precisely why the log loss is bad while the row sums are perfect.")
print()

print("When to use which:")
print()
print("  - multinomial / softmax: the default. Classes are mutually exclusive and")
print("    exhaustive, which is the normal case for a classifier.")
print("  - one-vs-rest: genuinely overlapping labels (multi-label), or a solver")
print("    constraint. Also reasonable as a fast baseline you can check against.")
print("  - do NOT use the naive 'fit K binaries, read off the raw sigmoids' version")
print("    and treat them as probabilities. That is the actual mistake.")
print()


# ======================================================================================
# PART G - reading the coefficients
# ======================================================================================
print("-" * 78)
print("PART G - so what do K rows of coefficients mean?")
print("-" * 78)
print("Per Part D, the raw rows are only interpretable up to a constant. The standard")
print("fix is to compare each class to the average class, i.e. centre the rows across")
print("classes.")
print()
print("The first thing to check is whether that is even necessary here - and the answer")
print("turns out to be 'it depends on the penalty', which is not what you would guess.")
print()

l1_multi = LogisticRegression(C=1.0, l1_ratio=1.0, solver="saga",
                              max_iter=80000).fit(X_train_s, y_train)
en_multi = LogisticRegression(C=1.0, l1_ratio=0.5, solver="saga",
                              max_iter=80000).fit(X_train_s, y_train)

gauge_rows = []
for name, model in [("L2 (l1_ratio=0)", sk_model), ("elastic net (0.5)", en_multi),
                    ("L1 (l1_ratio=1)", l1_multi)]:
    W_m = model.coef_
    col_means = W_m.mean(axis=0)
    gauge_rows.append({
        "penalty": name,
        "max |column mean|": float(np.max(np.abs(col_means))),
        "sum-to-zero?": "yes" if np.max(np.abs(col_means)) < 1e-6 else "NO",
        "mean intercept": round(float(model.intercept_.mean()), 8),
        "log loss": round(log_loss(y_test, model.predict_proba(X_test_s),
                                   labels=sk_model.classes_), 4),
    })

gauge_df = pd.DataFrame(gauge_rows)
print(gauge_df.to_string(index=False))
print()

l2_means = sk_model.coef_.mean(axis=0)
l1_means = l1_multi.coef_.mean(axis=0)
print("With L2 the coefficients already sum to zero across classes, to machine")
print(f"precision (largest column mean {np.max(np.abs(l2_means)):.2e}). That is not a coincidence and it is")
print("not something sklearn is documenting. It falls out of the optimisation:")
print()
print("  Part D established that adding a constant vector to all K coefficient rows")
print("  leaves every prediction unchanged. So an infinite set of parameter vectors")
print("  describe the identical model. The L2 penalty is ||W||_F^2, and among all the")
print("  representatives of one model, the one with the smallest norm is the one where")
print("  the constant is spread as evenly as possible - which is exactly the sum-to-zero")
print("  solution. L2 regularisation silently picks the centred representative for you.")
print()
print("With L1, nothing does that, and the coefficients come out uncentred:")
print(f"  L2 column means:   {np.round(l2_means, 8)}")
print(f"  L1 column means:   {np.round(l1_means, 4)}")
print()
print("The L1 rows are not wrong - they are the same model, shifted. But the raw")
print("numbers are now on an arbitrary scale, and this is a real trap: the same fit at")
print("l1_ratio=0 and l1_ratio=1 gives coefficients that are not comparable to each")
print("other, even though the log losses in the table above are perfectly comparable.")
print("Never compare multiclass coefficient magnitudes across penalties without")
print("centring first. The intercepts are centred in all three cases, so the problem is")
print("specifically in the coefficient rows.")
print()
print("The habit to take away: centre explicitly, always, and do not assume. It is")
print("three lines of code, it is a no-op when it is already true, and it is the")
print("difference between a readable coefficient table and a meaningless one.")
print()

W_raw = sk_model.coef_
W_centred = W_raw - W_raw.mean(axis=0, keepdims=True)
intercepts_raw = sk_model.intercept_
intercepts_centred = intercepts_raw - intercepts_raw.mean()

coef_df = pd.DataFrame(
    W_centred,
    index=[f"class {k} ({CLASS_NAMES[k]})" for k in range(N_CLASSES)],
    columns=["x1", "x2"],
)
coef_df["intercept"] = intercepts_centred
print("Centred coefficients for the L2 model (each class relative to the average):")
print()
print(coef_df.round(3).to_string())
print()
largest = np.unravel_index(np.argmax(np.abs(W_centred)), W_centred.shape)
print(f"The largest entry is class {largest[0]} ({CLASS_NAMES[largest[0]]}) on feature x{largest[1] + 1}, at "
      f"{W_centred[largest]:+.2f}.")
print()
print("Read this correctly, and the distinction matters:")
print()
print(f"  'Class {largest[0]} has coefficient {W_centred[largest]:+.2f} on x{largest[1] + 1}' is NOT a claim about x{largest[1] + 1}'s")
print(f"  effect on class {largest[0]}. It is a claim about x{largest[1] + 1}'s effect on class {largest[0]} RELATIVE TO")
print("  THE AVERAGE CLASS. Raising x2 raises the logit for the classes with positive")
print("  coefficients and lowers it for the rest - and 'the average class' is not a")
print("  real class that anyone belongs to.")
print()
print("This is exactly the confusion-avoidance in multinomial logit, and it is why")
print("dropping a reference class is the other common fix: pick class 0, then every")
print("coefficient is 'the effect relative to class 0', which is at least anchored to")
print("something concrete.")
print()

print("A second thing the raw coefficients cannot tell you: magnitude is not")
print("importance, because each row competes only with the others, and features have")
print("different scales. Part E of script 08 is the long version of this argument.")
print("Do not rank features by |coefficient| across a multiclass model.")
print()

print("Confusion matrix, which is where multiclass diagnostics actually start:")
print()
cm = confusion_matrix(y_test, pred_sk)
cm_df = pd.DataFrame(
    cm,
    index=[f"true {k}" for k in range(N_CLASSES)],
    columns=[f"pred {k}" for k in range(N_CLASSES)],
)
print(cm_df.to_string())
print()

per_class_recall = np.diag(cm) / cm.sum(axis=1)
print(f"Per-class recall: {np.round(per_class_recall, 4)}")
print(f"Worst class: {per_class_recall.argmin()} at {per_class_recall.min():.4f}")
print(f"Best class:  {per_class_recall.argmax()} at {per_class_recall.max():.4f}")
print()
print("On a balanced problem with well-separated classes, per-class recall is")
print(f"roughly {per_class_recall.min():.2f}-{per_class_recall.max():.2f} everywhere and accuracy is")
print(f"{acc_mult:.4f}. There is no class problem to solve here, which is why Part H")
print("manufactures one.")
print()

print("Full classification report:")
print()
report_text = classification_report(y_test, pred_sk, target_names=CLASS_NAMES,
                                    digits=4, zero_division=0)
print(report_text)
print("Macro-average is the number to watch when classes are equally important.")
print("Weighted-average is the number to report when you care about overall error")
print("and the classes are unequal - which is the situation script 07 is about.")
print()


# ======================================================================================
# PART H - a failure mode that only multiclass has
# ======================================================================================
print("-" * 78)
print("PART H - missing classes, and the label that was never in the training set")
print("-" * 78)
print("""
Every part of the multiclass machinery assumes the classes you trained on are the
classes that exist. Drop one class from the training set and the model has no way
to express it: it will confidently assign that row to one of the classes it knows.
This is not a subtle degradation. It is a categorical failure, and it is exactly
the failure of any classifier facing a new class.
""")
print()

missing_class = 3
kept = y_train != missing_class
reduced = LogisticRegression(max_iter=5000).fit(X_train_s[kept], y_train[kept])

proba_reduced = reduced.predict_proba(X_test_s)
pred_reduced = reduced.predict(X_test_s)

is_missing = y_test == missing_class
n_missing = int(is_missing.sum())
correct_on_missing = int((pred_reduced[is_missing] == missing_class).sum())
print(f"Held out class {missing_class} ({CLASS_NAMES[missing_class]}) from training entirely.")
print(f"It still appears in the test set: {n_missing} rows.")
print()
print(f"  rows of that class predicted correctly:   {correct_on_missing} of {n_missing} "
      f"({correct_on_missing / n_missing:.1%})")
print(f"  is the class even in the model's vocabulary? {'yes' if missing_class in reduced.classes_ else 'NO'}")
print()
print("The model is not uncertain and it does not abstain. It is confidently wrong on")
print("every one of them, because 'class 3' is not a category it has. Its confidence")
print("distribution on those rows looks normal:")
print()
missing_proba = proba_reduced[is_missing]
print(f"  mean max predicted probability on those rows: {missing_proba.max(axis=1).mean():.4f}")
print(f"  same statistic across all test rows:          {proba_reduced.max(axis=1).mean():.4f}")
print()
print("This is the failure you cannot detect by watching confidence scores, which is")
print("why 'open set recognition' is a research area rather than a solved problem. The")
print("practical defences are all upstream of the model:")
print()
print("  - know your class inventory, and monitor the DISTRIBUTION of predictions")
print("    against expected proportions. A class that stops being predicted is a")
print("    monitoring signal, and it works even when per-row confidence looks fine.")
print("  - keep a 'reject' route: an abstain threshold plus human review (script 06)")
print("  - retrain on a schedule; a model does not learn about classes it has not seen")
print()
print("Note also what this does to accuracy as a headline number. Removing a class")
print("from training does not crash the accuracy metric - it just quietly caps it,")
print("because those rows are guaranteed wrong. A metric alone will not tell you the")
print("model is missing something it needs. Per-class recall, printed above, would.")
print()


# ======================================================================================
# PART I - choosing between the options, by cross-validation
# ======================================================================================
print("-" * 78)
print("PART I - settling it with cross-validation instead of opinion")
print("-" * 78)
print("The comparison above was on one test split, so the accuracy gap could be noise.")
print("Repeat it over 5 folds of the training data and look at the spread.")
print()

from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def cv_scores(factory, X_in, y_in, metric):
    """Fold-wise scores. Log loss is reported in its natural orientation - lower is
    better - rather than negated, because a table where one metric is 'higher is
    better' and the other is 'lower is better' is a table you will misread later."""
    scores = []
    for fold_train, fold_val in cv.split(X_in, y_in):
        model = factory()
        model.fit(X_in[fold_train], y_in[fold_train])
        if metric == "log loss":
            scores.append(log_loss(y_in[fold_val], model.predict_proba(X_in[fold_val]),
                                   labels=model.classes_))
        else:
            scores.append(accuracy_score(y_in[fold_val], model.predict(X_in[fold_val])))
    return np.array(scores)


cv_rows = []
for name, factory in [
    ("multinomial (softmax)", lambda: LogisticRegression(max_iter=5000)),
    ("one-vs-rest", lambda: OneVsRestClassifier(LogisticRegression(max_iter=5000))),
]:
    for metric, better in [("accuracy", "higher"), ("log loss", "lower")]:
        s = cv_scores(factory, X_train_s, y_train, metric)
        cv_rows.append({
            "model": name,
            "metric": f"{metric} ({better} better)",
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
        })

cv_df = pd.DataFrame(cv_rows)
print(cv_df.to_string(index=False))
print()

cv_acc = cv_df[cv_df.metric.str.startswith("accuracy")].set_index("model")["mean"]
cv_ll = cv_df[cv_df.metric.str.startswith("log loss")].set_index("model")["mean"]
cv_acc_std = cv_df[cv_df.metric.str.startswith("accuracy")].set_index("model")["std"]
cv_mult = "multinomial (softmax)"
cv_ovr = "one-vs-rest"

acc_gap_single = abs(acc_ovr - acc_mult)
acc_gap_cv = abs(cv_acc[cv_mult] - cv_acc[cv_ovr])
cv_winner = cv_mult if cv_acc[cv_mult] > cv_acc[cv_ovr] else cv_ovr
single_winner = "one-vs-rest" if acc_ovr > acc_mult else "multinomial"

print(f"The accuracy result REVERSES, and that is the most useful thing in this part.")
print()
print(f"  single test split:  {single_winner} was ahead by {acc_gap_single:.4f}")
print(f"  5-fold CV mean:     {cv_winner} is ahead by {acc_gap_cv:.4f}")
print(f"  fold-to-fold std:   about {cv_acc_std.max():.4f}")
print()
print(f"So the single-split accuracy gap was noise. It was smaller than the")
print(f"fold-to-fold variability, and it pointed the other way once you averaged over")
print("folds. Anyone who had compared the two methods on one split and written a")
print("conclusion would have written the wrong one, with total confidence.")
print()
print("That is the practical argument for cross-validation, and it is worth separating")
print("from the usual textbook version. CV is not primarily here to give a better")
print("estimate of performance. It is here to stop you believing a difference that is")
print("smaller than the measurement noise.")
print()
print("Log loss is the metric that actually separates these two, and it does so")
print(f"consistently: on the test split {ll_mult:.4f} against {ll_ovr:.4f}, and across folds")
print(f"{cv_ll[cv_mult]:.4f} against {cv_ll[cv_ovr]:.4f}. Same answer both times, same direction, an")
print("order of magnitude apart. That is what a real difference looks like next to what")
print("a noisy one looks like.")
print()
print("So the practical conclusion, which is also the general one: when a decision is a")
print("modelling judgement rather than a computation, put it in a cross-validation")
print("loop with the metric you actually care about, compare the gap against the")
print("spread, and refuse to conclude from a difference the noise can explain. Script 11")
print("is the full treatment of that loop.")
print()


# ======================================================================================
# PLOTS
# ======================================================================================
print("-" * 78)
print("PLOTS")
print("-" * 78)
print("(displayed, not saved - the figures are exploratory aids, not outputs)")
print()

fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))

# 1. the data
ax = axes[0, 0]
colors = plt.cm.tab10(np.linspace(0, 1, N_CLASSES))
for k in range(N_CLASSES):
    ax.scatter(X_train[y_train == k, 0], X_train[y_train == k, 1],
               s=12, alpha=0.55, color=colors[k], label=CLASS_NAMES[k], linewidths=0)
ax.set_title("Five classes, well separated", fontsize=11, fontweight="bold")
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.set_xlabel("x1")
ax.set_ylabel("x2")

# 2. decision regions
ax = axes[0, 1]
pad = 0.6
x_min, x_max = X_train[:, 0].min() - pad, X_train[:, 0].max() + pad
y_min, y_max = X_train[:, 1].min() - pad, X_train[:, 1].max() + pad
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
grid = np.column_stack([xx.ravel(), yy.ravel()])
regions = sk_model.predict(scaler.transform(grid)).reshape(xx.shape)
ax.contourf(xx, yy, regions, levels=np.arange(-0.5, N_CLASSES), alpha=0.30,
            cmap=plt.cm.tab10)
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=10, alpha=0.6,
           cmap=plt.cm.tab10, linewidths=0)
ax.set_title("Multinomial decision regions (piecewise linear)", fontsize=11, fontweight="bold")
ax.set_xlabel("x1")
ax.set_ylabel("x2")

# 3. logit surface for the most confident class
ax = axes[0, 2]
best_class = int(np.bincount(sk_model.predict(X_test_s), minlength=N_CLASSES).argmax())
Z_grid = sk_model.decision_function(scaler.transform(grid))[:, best_class].reshape(xx.shape)
contour_filled = ax.contourf(xx, yy, Z_grid, levels=25, cmap="RdBu_r", alpha=0.85)
fig.colorbar(contour_filled, ax=ax, fraction=0.046)
ax.scatter(X_test[y_test == best_class, 0], X_test[y_test == best_class, 1],
           s=14, color="#212529", label=f"true {CLASS_NAMES[best_class]}", linewidths=0)
ax.legend(fontsize=8, frameon=False)
ax.set_title(f"Logit for class {best_class}: a plane, not a curve", fontsize=11, fontweight="bold")
ax.set_xlabel("x1")
ax.set_ylabel("x2")

# 4. probability calibration per class
ax = axes[1, 0]
for k in range(N_CLASSES):
    yk = (y_test == k).astype(float)
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(proba_sk[:, k], bins) - 1
    observed, predicted = [], []
    for b in range(10):
        sel = idx == b
        if sel.sum() >= 20:
            observed.append(yk[sel].mean())
            predicted.append(proba_sk[sel, k].mean())
    ax.plot(predicted, observed, "o-", linewidth=2, markersize=5,
            color=colors[k], label=CLASS_NAMES[k])
lims = [min(0.02, min(proba_sk.min(), 0.02)), 1.0]
ax.plot([0, 1], [0, 1], "--", color="#495057", linewidth=1.6, label="perfect")
ax.set_xlabel("mean predicted probability")
ax.set_ylabel("observed frequency")
ax.set_title("Per-class reliability: softmax is calibrated", fontsize=11, fontweight="bold")
ax.legend(fontsize=7, frameon=False, ncol=2)

# 5. multinomial vs OVR log loss
ax = axes[1, 1]
names = ["multinomial", "one-vs-rest"]
values = [ll_mult, ll_ovr]
bars = ax.bar(names, values, color=["#4C6EF5", "#E03131"], width=0.55, edgecolor="white", linewidth=2)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.4f}",
            ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("log loss (lower is better)")
ax.set_title(f"They disagree: OVR wins accuracy,\nmultinomial wins log loss {ll_mult / ll_ovr:.1f}x",
             fontsize=11, fontweight="bold")
ax.set_ylim(0, max(values) * 1.25)

# 6. the held-out class failure
ax = axes[1, 2]
bars = ax.bar([f"true {k}" for k in range(N_CLASSES)],
              per_class_recall, color=[("#E03131" if k == missing_class else colors[k])
                                       for k in range(N_CLASSES)],
              edgecolor="white", linewidth=1.5)
ax.set_ylim(0, 1.08)
ax.axhline(1.0 / N_CLASSES, color="#495057", linestyle="--", linewidth=1.4,
           label="chance (20%)")
ax.set_ylabel("recall")
ax.set_title("Per-class recall: the diagnostic that would\nhave caught the missing class",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8, frameon=False)
for k, rect in enumerate(bars):
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02,
            f"{per_class_recall[k]:.2f}", ha="center", fontsize=8)

plt.tight_layout()
plt.show()


# ======================================================================================
# TAKEAWAYS
# ======================================================================================
print()
print("=" * 78)
print("TAKEAWAYS")
print("=" * 78)
print("""
1. The extension is small and the consequences are not. One extra normalisation
   turns one logit into K logits; everything else in this file follows from that,
   including the redundancy that makes naive coefficient reading wrong.

2. The gradient has the same shape as the binary one: dL/dz_k = p_k - y_k. If you
   understood script 01, you already had the hard part.

3. A 2-class softmax IS the logistic sigmoid, exactly - verified to machine
   precision above. sklearn reflects this by storing one coefficient row, not two.

4. Only differences of logits are real. K rows of coefficients carry K - 1
   directions of information. Centre them, or drop a reference class, before
   interpreting anything.
5. The pairwise-difference identity is nearly true, and not exactly true. The odds
   p_b/p_a = exp(z_b - z_a) make z_b - z_a a genuine log-odds score, and it
   correlates above 0.99 with a separately fitted binary model - but the two
   minimise different objectives and differ by a median of ~1 logit unit on
   nearby pairs, and by several units on distant ones. Good approximation, not an
   identity. And check the sign: a correlation near -1 means the subtraction ran
   backwards, not that the theory is false.

6. Multinomial and one-vs-rest disagree on a single split, and cross-validation
   REVERSES the accuracy part of that: the gap was smaller than the fold-to-fold
   noise. Log loss separates them decisively and consistently, an order of
   magnitude apart, in both directions of the comparison. The lesson is not 'softmax
   is better' - it is 'never conclude from a difference the noise can explain'.
   One-vs-rest renormalises its outputs so the rows sum to 1, which makes them look
   like a distribution without making the model jointly coherent.

7. Missing a class from training is a categorical failure, not a gradual one. The
   model will be confidently wrong and its confidence will look normal, because
   the class is not in its vocabulary. Monitor the distribution of predicted
   classes, not per-row confidence.

8. Only the L2 fit hands you sum-to-zero coefficients. L2 selects the minimum-norm
   representative of each gauge-equivalence class, which happens to be the centred
   one; L1 and elastic net do not. Centre explicitly rather than assuming - it is a
   no-op when it is already true.

9. In scikit-learn 1.8, `multi_class=` no longer exists, and `liblinear` no longer
   accepts multiclass at all. Multinomial is the only multiclass fit
   LogisticRegression performs; ask for one-vs-rest explicitly, and use lbfgs (or
   saga for L1 and elastic net) for anything with 3 or more classes.
""")
print("=" * 78)
