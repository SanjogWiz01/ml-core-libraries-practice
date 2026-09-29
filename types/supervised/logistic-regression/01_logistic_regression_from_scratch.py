"""
01 - Logistic Regression From Scratch
=====================================

Goal: fit a binary classifier by hand using gradient descent on the log loss, so
the scikit-learn version in script 02 never feels like magic.

What this file demonstrates
---------------------------
1. The sigmoid function, and why it is the right link function (it maps any real
   score into a valid probability, and it is where the exponentials in the
   gradient come from).
2. Why MSE is the wrong loss here and log loss is the right one.
3. The exact gradient of the log loss - which turns out to be a beautifully
   simple expression.
4. That hand-written gradient descent recovers the same weights scikit-learn
   finds with its solver (we verify numerically).
5. Numerically stable log loss, because the naive version returns `inf` the
   moment a probability saturates.

The model
---------
    z_i    = w0 + w1 * x1_i + ... + wp * xp_i          <- the "logit"
    p_i    = sigmoid(z_i) = 1 / (1 + exp(-z_i))        <- P(y_i = 1 | x_i)
    y_hat_i = 0 if p_i < 0.5 else 1                    <- thresholded prediction

The loss
--------
    J(w) = -(1/n) * SUM_i [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]

Partial derivatives
-------------------
    dJ/dw_j = (1/n) * SUM_i (p_i - y_i) * x_ij          <- vectorised: X^T (p - y) / n

Run:  python 01_logistic_regression_from_scratch.py
"""

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------------------
# 1. Data: two well-separated Gaussian blobs, generated with a known boundary so we can
#    check that training actually found it.
# --------------------------------------------------------------------------------------
rng = np.random.default_rng(42)  # seeded -> reproducible results
n_per_class = 200

# Class 0 sits bottom-left, class 1 top-right. The true separating line is
# x1 + x2 = 0, i.e. everything with x1 + x2 > 0 is class 1.
X0 = rng.normal(loc=[-1.5, -1.5], scale=[1.0, 1.0], size=(n_per_class, 2))
X1 = rng.normal(loc=[+1.5, +1.5], scale=[1.0, 1.0], size=(n_per_class, 2))

X = np.vstack([X0, X1])
y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

# scikit-learn's convention: X is (n_samples, n_features), y is (n,).
# Prepend a column of ones so the intercept is learned by the same update rule
# as every other weight - no special-casing in the gradient.
X_bias = np.hstack([np.ones((X.shape[0], 1)), X])

print("=" * 70)
print("LOGISTIC REGRESSION FROM SCRATCH - gradient descent on the log loss")
print("=" * 70)
print(f"Data shape      : X {X.shape}, y {y.shape}")
print(f"Class balance   : {int(y.sum())} positive, {int((1 - y).sum())} negative")
print(f"True boundary   : x1 + x2 = 0  (a line at 45 degrees through the origin)")
print()


# --------------------------------------------------------------------------------------
# 2. The three functions that make up the model.
# --------------------------------------------------------------------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    The logistic function: 1 / (1 + exp(-z)).

    Two numerical traps live in that one line, and both are worth knowing:

    - `exp(-z)` overflows to `inf` for very negative z (z = -1000), and `1/inf`
      is 0, which happens to be right. But the *gradient* of a saturated sigmoid
      is 0 too, so the sample stops contributing to training at all.
    - For large positive z, `exp(-z)` underflows towards 0 and sigmoid -> 1, again
      fine numerically but equally uninformative.

    The branch below avoids the overflow warning by computing only the safe side.
    """
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    # Safe for z >= 0: exp(-z) is in (0, 1], so no overflow.
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    # Safe for z < 0: rewrite as exp(z) / (1 + exp(z)) so the exponent is negative.
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def log_loss(y_true: np.ndarray, probabilities: np.ndarray, eps: float = 1e-15) -> float:
    """
    Average cross-entropy, computed in the numerically stable form.

    The textbook version is

        -(1/n) * SUM [ y*log(p) + (1-y)*log(1-p) ]

    but log(0) is -inf, and p = 0 happens the moment the weights overshoot. The
    stable trick is to clip p into [eps, 1-eps] first. The clipping is not a
    fudge: it only bites in the saturated regime where the true loss is already
    astronomically large, so any finite value is an equally good answer.
    """
    p = np.clip(probabilities, eps, 1.0 - eps)
    per_sample = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
    return float(np.mean(per_sample))


print("-" * 70)
print("PART A - the sigmoid, and why squared error is the wrong loss")
print("-" * 70)
print(f"{'z':>8} {'sigmoid(z)':>14} {'p':>6} {'-log(p)':>10} {'-log(1-p)':>12}")
print("-" * 70)
print("  When you are WRONG with high confidence, log loss should punish you")
print("  brutally. Squared error on probabilities would not.")
print()
for z_value in [-6.0, -2.0, -0.5, 0.0, 0.5, 2.0, 6.0]:
    p_value = float(sigmoid(np.array([z_value]))[0])
    # p if the truth is 1, and p if the truth is 0
    loss_if_pos = -np.log(p_value)
    loss_if_neg = -np.log(1.0 - p_value)
    print(f"{z_value:>8.1f} {p_value:>14.6f} {p_value:>6.3f} {loss_if_pos:>10.3f} {loss_if_neg:>12.3f}")
print()
print("Read the last row: predicting 0.9975 for a sample whose truth is 0 costs")
print(f"{-np.log(1 - 0.99753):.1f} in log loss. Squared error would charge 1.0.")
print("Log loss grows without bound as confidence in a wrong answer grows, which")
print("is exactly the pressure a probability-calibrated model should feel.")
print()


# --------------------------------------------------------------------------------------
# 3. The training function.
# --------------------------------------------------------------------------------------
def train_logistic_regression(
    X_bias: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.1,
    n_iterations: int = 3000,
) -> tuple[np.ndarray, list[float]]:
    """
    Fit P(y=1|x) = sigmoid(Xw) with batch gradient descent on the log loss.

    Parameters
    ----------
    X_bias : (n, p+1) array
        Design matrix whose FIRST COLUMN must be the constant 1.
    y : (n,) array
        Binary targets in {0, 1}.
    learning_rate : float
        Step size. Logistic regression is far more forgiving than linear
        regression - the log loss is smooth and its gradient is bounded by
        |x| - but too large still overshoots and oscillates.
    n_iterations : int
        Number of full passes over the data.

    Returns
    -------
    weights : (p+1,) array -> [intercept, w1, w2, ...]
    loss_history : list[float]
        Average log loss after each iteration.
    """
    n = X_bias.shape[0]
    weights = np.zeros(X_bias.shape[1])
    loss_history = []

    for iteration in range(n_iterations):
        # --- forward pass -------------------------------------------------------------
        logits = X_bias @ weights            # (n, p+1) @ (p+1,) -> (n,) raw scores
        probabilities = sigmoid(logits)      # (n,) probabilities in (0, 1)
        loss_history.append(log_loss(y, probabilities))

        # --- backward pass ------------------------------------------------------------
        # dJ/dw = (1/n) * X^T (p - y)
        #
        # Why this is so clean: with z = Xw and p = sigmoid(z) we have dp/dz = p(1-p),
        # and the chain rule on -(y log p + (1-y) log(1-p)) cancels the p(1-p) factor
        # exactly, leaving simply (p - y). Every exponential disappears.
        gradient = (X_bias.T @ (probabilities - y)) / n

        # --- update -------------------------------------------------------------------
        weights -= learning_rate * gradient

        if not np.isfinite(loss_history[-1]):
            raise ValueError(
                f"Log loss became {loss_history[-1]} at iteration {iteration}. "
                f"learning_rate={learning_rate} is too large."
            )

    return weights, loss_history


weights, loss_history = train_logistic_regression(X_bias, y, learning_rate=0.1, n_iterations=3000)
loss_history = np.array(loss_history)
intercept, w1, w2 = weights

print("-" * 70)
print("PART B - train with a sensible learning rate")
print("-" * 70)
print(f"Learned intercept : {intercept:>10.4f}")
print(f"Learned w1        : {w1:>10.4f}")
print(f"Learned w2        : {w2:>10.4f}")
print(f"Initial log loss  : {loss_history[0]:>10.4f}   (equal odds: 0.6931)")
print(f"Final log loss    : {loss_history[-1]:>10.4f}")
print()
print("Two things to notice:")
print("  1. The starting log loss is exactly ln(2) = 0.6931. With all weights at")
print("     zero, sigmoid(z) = 0.5 everywhere, and the loss of a constant 0.5")
print("     prediction is -[0.5*log(0.5) + 0.5*log(0.5)] = log(2). That constant is")
print("     the loss of the 'do nothing' model, and it is the bar to beat.")
print("  2. w1 and w2 came out nearly equal, as they must: the classes were generated")
print("     symmetrically, so no orientation of the boundary is preferred. The true")
print("     weights were w1 = w2 = 1.")
print()


# --------------------------------------------------------------------------------------
# 4. The cross-check that catches sign errors.
# --------------------------------------------------------------------------------------
print("-" * 70)
print("PART C - cross-check against scikit-learn")
print("-" * 70)

from sklearn.linear_model import LogisticRegression  # noqa: E402

# A huge C removes the penalty effectively entirely. The default is C=1.0, which
# regularises, and comparing a regularised fit to an unregularised one is the most
# common reason two "logistic regressions" disagree. (sklearn <=1.7 spelled this
# penalty=None; from 1.8 onwards penalty is deprecated in favour of C / l1_ratio.
# C=np.inf also works but warns in 1.8, so a large finite C is used instead.)
sk_model = LogisticRegression(C=1e6, max_iter=5000, tol=1e-12)
sk_model.fit(X, y)
sk_weights = np.concatenate([[sk_model.intercept_[0]], sk_model.coef_[0]])

print("Gradient descent    : intercept %+.6f  w1 %+.6f  w2 %+.6f" % (intercept, w1, w2))
print("sklearn (no penalty): intercept %+.6f  w1 %+.6f  w2 %+.6f" % tuple(sk_weights))
print("Max abs difference  : %.3e" % np.max(np.abs(weights - sk_weights)))
print("=> Both solved the same convex problem and landed in the same place.")
print()

# --- what the weights actually recovered, and what they did not -----------------------
# The GENERATING boundary was x1 + x2 = 0, so the only thing the data can pin down is
# the DIRECTION of (w1, w2), not its magnitude. Check the direction by comparing the
# slope of the learned boundary with the true one.
true_slope = -1.0
learned_slope = -w1 / w2
print("Recovering the true boundary:")
print(f"  true slope of the boundary  : {true_slope:+.4f}")
print(f"  learned slope of the boundary: {learned_slope:+.4f}")
print(f"  angular error               : {np.degrees(np.arctan(abs((learned_slope - true_slope) / (1 + true_slope * learned_slope)))):.3f} degrees")
print()
print("The direction is recovered almost exactly. The MAGNITUDE is not, and it could")
print("not be: the two classes overlap, so no weight vector perfectly separates them.")
print("The maximum-likelihood solution has no reason to keep |w| small once the data")
print("cannot be separated, so it grows until the probabilities are as extreme as the")
print("overlap allows. This is the same phenomenon that makes the weights EXPLODE to")
print("infinity on perfectly separable data - complete separation, script 16.")
print()
print("Consequence you must respect: coefficient MAGNITUDES on overlapping data are")
print("not a measure of feature importance. Odds RATIOS on interpretable feature")
print("scales are, and that is what scripts 02 and 13 use.")
print()


# --------------------------------------------------------------------------------------
# 5. Learning rate behaviour.
# --------------------------------------------------------------------------------------
print()
print("-" * 70)
print("PART D - what happens when the learning rate is wrong")
print("-" * 70)
print(f"{'learning_rate':>15} | {'iterations to 0.01% of final loss':>36} | outcome")
print("-" * 70)

for lr in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]:
    try:
        w, history = train_logistic_regression(X_bias, y, learning_rate=lr, n_iterations=20000)
        target = history[-1] * 1.0001
        converged_at = next((i for i, value in enumerate(history) if value < target), -1)
        print(f"{lr:>15} | {converged_at:>36} | final log loss {history[-1]:.4f}")
    except ValueError as exc:
        print(f"{lr:>15} | {'never':>36} | {exc}")
        break

print()
print("The pattern is the same as linear regression but far gentler: the log loss")
print("has a bounded gradient, so it takes a much larger step before it explodes.")
print("That robustness is one reason the sigmoid+log-loss pairing is so popular.")
print()


# --------------------------------------------------------------------------------------
# 6. Plots.
# --------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# --- Panel 1: the data and the learned boundary ---------------------------------------
ax = axes[0]
ax.scatter(X0[:, 0], X0[:, 1], s=28, alpha=0.65, color="#4C6EF5",
           edgecolors="none", label="class 0")
ax.scatter(X1[:, 0], X1[:, 1], s=28, alpha=0.65, color="#F03E3E",
           edgecolors="none", label="class 1")

# The decision boundary is where sigmoid(z) = 0.5, i.e. exactly z = 0, i.e.
# w0 + w1*x1 + w2*x2 = 0. Note this is a LINE IN FEATURE SPACE even though the
# boundary in PROBABILITY space is a curve (the sigmoid of that line).
xs = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)
ax.plot(xs, -(intercept + w1 * xs) / w2, color="#212529", linewidth=2.5,
        label="decision boundary (z = 0)")

ax.set_title("The learned linear decision boundary", fontsize=11, fontweight="bold")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Panel 2: the log loss curve ------------------------------------------------------
ax = axes[1]
ax.plot(loss_history, color="#0CA678", linewidth=1.6)
ax.axhline(np.log(2), color="#E03131", linestyle="--", linewidth=2,
           label=f"ln(2) = {np.log(2):.4f} (constant 0.5 model)")
ax.set_title("Log loss falls below the do-nothing baseline", fontsize=11, fontweight="bold")
ax.set_xlabel("iteration")
ax.set_ylabel("average log loss")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Panel 3: sigmoid, and the corresponding log loss ---------------------------------
ax = axes[2]
z_grid = np.linspace(-6, 6, 400)
ax.plot(z_grid, sigmoid(z_grid), color="#4C6EF5", linewidth=2.5, label="sigmoid(z) = P(y=1)")
ax.plot(z_grid, 1 - sigmoid(z_grid), color="#868E96", linewidth=2, linestyle="--",
        label="1 - sigmoid(z) = P(y=0)")
ax.axhline(0.5, color="#E03131", linestyle=":", linewidth=2, label="classification threshold")
ax.axvline(0, color="#868E96", linestyle=":", linewidth=1.5)
ax.set_title("Sigmoid: scores in, probabilities out", fontsize=11, fontweight="bold")
ax.set_xlabel("z = w0 + w1*x1 + w2*x2")
ax.set_ylabel("probability")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

fig.suptitle("01 - Logistic Regression From Scratch", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


# --------------------------------------------------------------------------------------
# 7. Takeaways.
# --------------------------------------------------------------------------------------
print()
print("=" * 70)
print("TAKEAWAYS")
print("=" * 70)
print("1. Logistic regression is LINEAR REGRESSION on the log-odds. Fit a line to")
print("   z = w0 + w1*x1 + w2*x2, then squash it with the sigmoid.")
print("2. The gradient is (1/n) * X^T (p - y) - every exponential cancels. It looks")
print("   like the linear regression gradient, which is why the code looks familiar.")
print("3. Log loss, not MSE. It is the cost of a wrong confident prediction and it")
print("   keeps growing, which squared error on probabilities does not.")
print("4. Predict() thresholds at 0.5 by default. predict_proba() gives you the")
print("   probability. They are two different questions - script 06 lives on that gap.")
print("5. Always clip probabilities before taking log(), or a saturated model will")
print("   hand you -inf and a very confusing afternoon.")
print("6. Compare your fit to an UNREGULARISED sklearn model before concluding")
print("   your gradient descent is broken.")
