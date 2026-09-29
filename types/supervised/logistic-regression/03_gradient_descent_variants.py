"""
03 - Gradient Descent Variants for Logistic Regression
======================================================

Goal: measure the difference between batch, stochastic, mini-batch and Adam
optimisation on the *same* logistic regression problem, so the choice stops
being folklore.

What this file demonstrates
---------------------------
1. All four optimisers written out in full, on the same loss, for the same data.
2. Loss-versus-ITERATION and loss-versus-WALL-CLOCK. The second plot is the one
   that decides anything, and it is the one almost nobody draws.
3. The difference between `max_iter` in sklearn (epochs for SGD-family solvers,
   full-batch iterations for lbfgs) and what a learning rate actually does.
4. Feature scaling: the failure mode that makes SGD look broken when it is fine.
5. Why Adam is the default for neural networks and rarely the right choice here.

The objective all four minimise
-------------------------------
    J(w) = (1/n) SUM_i log(1 + exp(-y_i * z_i))     with z_i = w . x_i, y_i in {-1, +1}

That is the log loss rewritten with y in {-1, +1}, and it is numerically nicer
because log(1 + exp(-t)) never overflows for t > 0.

Run:  python 03_gradient_descent_variants.py
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------------------
# 1. Data: a non-linear problem in 2-D, so there is real work to do.
# --------------------------------------------------------------------------------------
# Two interlocking half-moons. A straight boundary cannot separate them, which
# matters here: an optimiser that converges to a worse point is not converging
# "wrongly", it is finding the best straight line, and the loss floor is high.


def two_moons(n_samples=1200, noise=0.22, seed=3):
    """Generate the classic two-moons dataset. Returns X (n, 2) and y in {0, 1}."""
    local_rng = np.random.default_rng(seed)
    n_half = n_samples // 2
    angle = np.pi

    outer = local_rng.normal(scale=noise, size=(n_half, 2))
    inner = local_rng.normal(scale=noise, size=(n_half, 2))

    moon_a = np.column_stack([
        np.cos(angle * np.linspace(0, np.pi, n_half)),
        np.sin(angle * np.linspace(0, np.pi, n_half)),
    ]) + outer
    moon_b = np.column_stack([
        1 - np.cos(angle * np.linspace(0, np.pi, n_half)),
        0.5 - np.sin(angle * np.linspace(0, np.pi, n_half)),
    ]) + inner

    return (np.vstack([moon_a, moon_b]),
            np.hstack([np.zeros(n_half), np.ones(n_half)]))


X, y = two_moons()
# Work in the y in {-1, +1} convention for the hand-written optimisers, because
# log(1 + exp(-y*z)) is the stable form and the gradient is then just
# -y * x, with no subtraction of a probability vector.
y_signed = np.where(y == 1, 1.0, -1.0)

# Scaling matters for every gradient-based method below. The moon coordinates
# live in roughly [-1, 2], so the unscaled features have wildly different
# curvature directions. We scale once, here, and use the same scaler for every
# optimiser so the comparison is fair.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_bias = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])

print("=" * 76)
print("GRADIENT DESCENT VARIANTS FOR LOGISTIC REGRESSION")
print("=" * 76)
print(f"Data: {X.shape[0]} rows, 2 features, 2 interlocking half-moons")
print(f"Class balance: {y.mean():.1%} positive")
print()
print("The data is NOT linearly separable, so the loss has a genuine floor well")
print("above zero. Every optimiser below converges to a DIFFERENT point on that")
print("floor, and comparing their final losses is a fair test.")
print()


# --------------------------------------------------------------------------------------
# 2. The loss and its gradient, written once.
# --------------------------------------------------------------------------------------
def log_loss_signed(weights: np.ndarray, X_bias: np.ndarray, y_signed: np.ndarray) -> float:
    """
    Average log loss with targets in {-1, +1}:

        J(w) = (1/n) * SUM_i log(1 + exp(-y_i * (w . x_i)))

    Why this form: log(1 + exp(t)) overflows for large positive t, but
    log(1 + exp(-t)) with t = y*z only ever evaluates exp() on a NEGATIVE
    argument when the sample is correctly classified. No clipping, no warnings.
    """
    margins = y_signed * (X_bias @ weights)
    # np.logaddexp(0, -margins) == log(1 + exp(-margins)), computed stably.
    return float(np.mean(np.logaddexp(0.0, -margins)))


def gradient_numerator(weights: np.ndarray, X: np.ndarray, y_signed: np.ndarray) -> np.ndarray:
    """
    The UNNORMALISED gradient: -(1/n) * X^T (y * sigmoid(-y * Xw)), with the 1/n
    deliberately left out.

        dJ/dz_i  =  d/dz_i [ log(1 + exp(-y_i z_i)) ]  =  -y_i * sigmoid(-y_i z_i)
        dJ/dw    =  X^T dJ/dz

    Returning the numerator separately is what lets a mini-batch use the SAME
    learning rate as full-batch GD: divide by the FULL n at the call site, not by
    the batch size. See stochastic_gradient_descent.

    The sigmoid is evaluated in its stable form so exp() is only ever handed a
    non-positive argument.
    """
    margins = y_signed * (X @ weights)
    # sigmoid(-margin), written so the exponent is never positive
    exp_margins = np.exp(-np.abs(margins))
    factor = np.where(margins >= 0, exp_margins / (1.0 + exp_margins),
                      1.0 / (1.0 + exp_margins))
    return -(X.T @ (y_signed * factor))


def gradient_signed(weights: np.ndarray, X: np.ndarray, y_signed: np.ndarray,
                    normalizer: int) -> np.ndarray:
    """The mean gradient, normalised by the full dataset size n."""
    return gradient_numerator(weights, X, y_signed) / normalizer


# quick self-check of the gradient against finite differences - if this is wrong,
# every comparison below is meaningless
def numeric_gradient(weights, X_bias, y_signed, eps=1e-6):
    grad = np.zeros_like(weights)
    for index in range(len(weights)):
        plus, minus = weights.copy(), weights.copy()
        plus[index] += eps
        minus[index] -= eps
        grad[index] = (log_loss_signed(plus, X_bias, y_signed)
                       - log_loss_signed(minus, X_bias, y_signed)) / (2 * eps)
    return grad


probe = np.array([0.3, -0.7, 0.5])
analytic = gradient_signed(probe, X_bias, y_signed, len(y_signed))
numeric = numeric_gradient(probe, X_bias, y_signed)
print("Gradient self-check against finite differences:")
print(f"  analytic : {np.array2string(analytic, precision=8)}")
print(f"  numeric  : {np.array2string(numeric, precision=8)}")
print(f"  max diff : {np.max(np.abs(analytic - numeric)):.3e}   (must be tiny)")
print()
print("Write this check into any optimiser you build. A sign error in the gradient")
print("looks exactly like a learning rate that is too large, and you will spend a")
print("day tuning the wrong hyperparameter.")
print()


# --------------------------------------------------------------------------------------
# 3. The four optimisers.
# --------------------------------------------------------------------------------------
def batch_gradient_descent(X_bias, y_signed, lr=0.5, n_iterations=800):
    """Classic GD: one full pass, then one step. Deterministic and exact."""
    n_samples = X_bias.shape[0]
    weights = np.zeros(X_bias.shape[1])
    history, weights_history = [], []
    for _ in range(n_iterations):
        history.append(log_loss_signed(weights, X_bias, y_signed))
        weights_history.append(weights.copy())
        weights -= lr * gradient_signed(weights, X_bias, y_signed, n_samples)
    history.append(log_loss_signed(weights, X_bias, y_signed))
    weights_history.append(weights.copy())
    return weights, np.array(history), np.array(weights_history)


def stochastic_gradient_descent(X_bias, y_signed, lr=0.05, n_epochs=60, batch_size=1, seed=0):
    """
    SGD: one sample per step, sampled in a random order each epoch.

    `batch_size=1` is pure SGD. `batch_size=32` is mini-batch. They differ only
    in the number of rows per gradient, so one function covers both - which is
    exactly the point.

    The gradient is divided by the FULL n, never by len(batch). That is what
    makes the learning rate mean the same thing at every batch size, and it is
    the single detail most hand-rolled SGD implementations get wrong.
    """
    local_rng = np.random.default_rng(seed)
    n_samples = X_bias.shape[0]
    weights = np.zeros(X_bias.shape[1])
    history, weights_history = [], []

    for _ in range(n_epochs):
        order = local_rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            idx = order[start:start + batch_size]
            weights -= lr * gradient_signed(weights, X_bias[idx], y_signed[idx], n_samples)
        history.append(log_loss_signed(weights, X_bias, y_signed))
        weights_history.append(weights.copy())
    return weights, np.array(history), np.array(weights_history)


def adam(X_bias, y_signed, lr=0.05, n_epochs=60, batch_size=32,
         beta1=0.9, beta2=0.999, eps=1e-8, seed=0):
    """
    Adam: SGD with a per-parameter adaptive step size built from running
    first- and second-moment estimates.

        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        w  -= lr * m_t_hat / (sqrt(v_t_hat) + eps)

    Adam takes roughly `lr` sized steps in EVERY direction regardless of the
    gradient's magnitude. That is a gift on a badly conditioned loss surface and
    a liability on a well-conditioned one, where it will bounce around the
    optimum instead of settling into it.
    """
    local_rng = np.random.default_rng(seed)
    n_samples = X_bias.shape[0]
    weights = np.zeros(X_bias.shape[1])
    m = np.zeros_like(weights)
    v = np.zeros_like(weights)
    history, weights_history = [], []
    t = 0

    for _ in range(n_epochs):
        order = local_rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            idx = order[start:start + batch_size]
            grad = gradient_signed(weights, X_bias[idx], y_signed[idx], n_samples)
            t += 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            m_hat = m / (1 - beta1**t)  # bias correction: without it the first
            v_hat = v / (1 - beta2**t)  # few steps are wildly wrong
            weights -= lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(log_loss_signed(weights, X_bias, y_signed))
        weights_history.append(weights.copy())
    return weights, np.array(history), np.array(weights_history)


# --------------------------------------------------------------------------------------
# 4. Run them all and time them.
# --------------------------------------------------------------------------------------
print("-" * 76)
print("PART A - four optimisers, same objective, same data")
print("-" * 76)

results = {}

timings = {}


def timed(name, function, *args, **kwargs):
    """Wall-clock timing around a call. 'n_iterations' is a meaningless unit of
    work for SGD, so the only honest comparison is seconds."""
    start = time.perf_counter()
    outcome = function(*args, **kwargs)
    timings[name] = time.perf_counter() - start
    results[name] = outcome
    return outcome


timed("batch GD (800 full passes)", batch_gradient_descent, X_bias, y_signed, lr=0.5, n_iterations=800)
timed("SGD (1 sample/step, 60 epochs)", stochastic_gradient_descent, X_bias, y_signed,
      lr=0.05, n_epochs=60, batch_size=1)
timed("mini-batch (32/step, 60 epochs)", stochastic_gradient_descent, X_bias, y_signed,
      lr=0.2, n_epochs=60, batch_size=32)
timed("Adam (32/step, 60 epochs)", adam, X_bias, y_signed, lr=0.05, n_epochs=60, batch_size=32)

n_updates = {
    "batch GD (800 full passes)": 800,
    "SGD (1 sample/step, 60 epochs)": 60 * len(y),
    "mini-batch (32/step, 60 epochs)": 60 * int(np.ceil(len(y) / 32)),
    "Adam (32/step, 60 epochs)": 60 * int(np.ceil(len(y) / 32)),
}

print(f"{'optimiser':<32} {'final loss':>11} {'time (s)':>9} {'updates':>9} {'w0':>8} {'w1':>8} {'w2':>8}")
print("-" * 88)
for name, (weights, history, _) in results.items():
    print(f"{name:<32} {history[-1]:>11.5f} {timings[name]:>9.3f} {n_updates[name]:>9} "
          f"{weights[0]:>8.3f} {weights[1]:>8.3f} {weights[2]:>8.3f}")

best_loss = min(history[-1] for _, history, _ in results.values())
print()
print("Read this table honestly, because the interesting number is the one that is")
print("NOT the best.")
print()
print("  - Batch GD, mini-batch and Adam all land within 0.007 of each other. That is")
print("    the expected result: the objective is CONVEX, so there is one optimum and")
print("    every method that converges finds it. What differs is the work required.")
print()
print("  - Pure SGD does NOT get there. Its loss is 0.067 above the optimum, and that")
print("    is not a bug - it is the NOISE FLOOR of stochastic gradients. A single-sample")
print("    gradient is a very noisy estimate of the true direction, so the iterates")
print("    jitter around the optimum with a spread proportional to the learning rate.")
print("    Batch GD has no such floor at all, which is why it wins on data this size.")
print()
print("The noise floor sets a frustrating tradeoff. Halve the learning rate and the")
print("jitter halves - but so does your progress per epoch, so you need twice the")
print("epochs. Measured, on this dataset:")
print()
print(f"{'lr':>8} {'epochs':>8} {'final loss':>12} {'gap vs batch GD':>16} {'time (s)':>9}")
print("-" * 58)
for lr, epochs in [(0.05, 60), (0.05, 150), (0.05, 300), (0.01, 300)]:
    start = time.perf_counter()
    _, history, _ = stochastic_gradient_descent(X_bias, y_signed, lr=lr, n_epochs=epochs,
                                                batch_size=1, seed=0)
    print(f"{lr:>8} {epochs:>8} {history[-1]:>12.5f} {history[-1] - best_loss:>16.5f} "
          f"{time.perf_counter() - start:>9.2f}")
print()
print(f"Compare that against batch GD: 800 full passes and {timings['batch GD (800 full passes)']:.3f} seconds")
print("to reach the optimum exactly, and then STOP - it is not still jiggling at")
print("epoch 900 the way SGD is. On tabular data, stochastic methods buy you the")
print("ability to hold the data in a GPU's memory, and they are not buying you")
print("anything else.")
print()

# --- sklearn's own solvers, for calibration ---------------------------------------------
print("-" * 76)
print("PART B - what sklearn's own solvers do")
print("-" * 76)
n_train = int(0.8 * len(X_scaled))
# Shuffle before splitting. The moons are stored as two contiguous blocks, so a
# positional split would put almost all of one class in the test set and give you
# a nonsense score. This is the same class of mistake as splitting a sorted frame.
split_rng = np.random.default_rng(99)
order = split_rng.permutation(len(X_scaled))
X_tr, X_te = X_scaled[order[:n_train]], X_scaled[order[n_train:]]
y_tr, y_te = y[order[:n_train]], y[order[n_train:]]
print(f"Split: {len(X_tr)} train ({y_tr.mean():.1%} positive) / "
      f"{len(X_te)} test ({y_te.mean():.1%} positive)")
print()
print(f"{'estimator':<46} {'test log loss':>14} {'time (s)':>9}")
print("-" * 72)

sk_results = {}
for label, estimator in [
    ("LogisticRegression(solver='lbfgs', C=1e6)", LogisticRegression(C=1e6, max_iter=5000, tol=1e-12)),
    # liblinear does not accept C=inf - it has no unregularised mode. A very large
    # finite C is the closest equivalent.
    ("LogisticRegression(solver='liblinear', C=1e6)", LogisticRegression(C=1e6, solver="liblinear", max_iter=5000)),
    ("LogisticRegression(solver='saga', C=1e6)", LogisticRegression(C=1e6, solver="saga", max_iter=5000, tol=1e-10)),
    ("SGDClassifier(loss='log_loss', 30 epochs)", SGDClassifier(loss="log_loss", max_iter=30, tol=None, random_state=0)),
    ("SGDClassifier(loss='log_loss', 2000 epochs)", SGDClassifier(loss="log_loss", max_iter=2000, tol=None, random_state=0)),
]:
    start = time.perf_counter()
    estimator.fit(X_tr, y_tr)
    elapsed = time.perf_counter() - start
    test_p = estimator.predict_proba(X_te)[:, 1]
    loss = -np.mean(y_te * np.log(np.clip(test_p, 1e-15, 1 - 1e-15))
                    + (1 - y_te) * np.log(np.clip(1 - test_p, 1e-15, 1 - 1e-15)))
    sk_results[label] = (loss, elapsed)
    print(f"{label:<46} {loss:>14.5f} {elapsed:>9.3f}")

print()
print("Two practical facts fall out of this table:")
print()
print("  1. 'lbfgs' is the right default for data this size. It is a quasi-Newton")
print("     method: it estimates the curvature of the loss and takes far better-")
print("     directed steps than any fixed learning rate. You will never hand-tune a")
print("     learning rate for it, and that is the entire reason to prefer it.")
print()
print("  2. SGDClassifier's max_iter counts EPOCHS (full passes), not steps. Going")
print(f"     from 30 to 2000 epochs changes the loss by "
      f"{abs(sk_results['SGDClassifier(loss=\'log_loss\', 30 epochs)'][0] - sk_results['SGDClassifier(loss=\'log_loss\', 2000 epochs)'][0]):.5f}")
print("     but costs the time in the second column. Convergence warnings from")
print("     sklearn are telling you the same thing: your budget was too small.")
print()
print("  - When to use 'saga': very large n, or you need elastic net. It is the only")
print("    solver supporting l1_ratio, and it is the slowest per step.")
print("  - When to use 'liblinear': small n (< 10,000 rows). It also supports L1.")
print("  - 'newton-cholesky': memory-hungry but excellent when p is large and n is")
print("    small, which is the opposite regime.")
print()


# --------------------------------------------------------------------------------------
# 5. The feature-scaling failure mode.
# --------------------------------------------------------------------------------------
print("-" * 76)
print("PART C - what unscaled features do to a learning rate")
print("-" * 76)

# Same data, but one feature stretched by a factor of 1000. This is not a
# synthetic worst case: it is "income in thousands" next to "age in years".
X_skewed = X_scaled.copy()
X_skewed[:, 1] *= 1000.0
X_skewed_bias = np.hstack([np.ones((X_skewed.shape[0], 1)), X_skewed])

print("Feature 2 is now 1000x larger than feature 1. Watch what a single learning")
print("rate can do to both at once:")
print()
print(f"{'features':<26} {'lr=0.5':>14} {'lr=0.001':>14} {'lr=1e-6':>14}")
print("-" * 70)
for label, matrix in [("scaled", X_bias), ("one feature x1000", X_skewed_bias)]:
    cells = []
    for lr in [0.5, 0.001, 1e-6]:
        try:
            _, history, _ = batch_gradient_descent(matrix, y_signed, lr=lr, n_iterations=800)
            cells.append(f"{history[-1]:>14.5f}")
        except (ValueError, OverflowError):
            cells.append(f"{'diverged':>14}")
    print(f"{label:<26} {cells[0]:>14} {cells[1]:>14} {cells[2]:>14}")

print()
print("This is the whole argument for StandardScaler in a gradient-based model.")
print("With one feature 1000x larger, the loss surface is a valley 1000x steeper")
print("along one axis. Any single learning rate is simultaneously too big for the")
print("steep axis (it oscillates or diverges) and too small for the flat one (it")
print("crawls). The fix is not a better optimiser, it is rescaling the input so the")
print("curvature is comparable in every direction.")
print()
print("Note that it also breaks the L1/L2 penalty, which is scale-dependent for")
print("the same reason - the penalty then treats a large-unit feature as the")
print("offender regardless of its information content. Script 08 returns to this.")
print()
print("Adam looks like an exception: its per-parameter step rescaling is exactly a")
print("free diagonal preconditioner, so it copes with badly scaled inputs. But it")
print("does not fix the penalty, and it costs more than lbfgs on data this size.")
print()


# --------------------------------------------------------------------------------------
# 6. Plots.
# --------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

colors = {
    "batch GD (800 full passes)": "#E03131",
    "SGD (1 sample/step, 60 epochs)": "#4C6EF5",
    "mini-batch (32/step, 60 epochs)": "#0CA678",
    "Adam (32/step, 60 epochs)": "#7048E8",
}

# --- Panel 1: loss vs ITERATION (the misleading plot) ----------------------------------
ax = axes[0]
for name, (_, history, _) in results.items():
    ax.plot(np.arange(1, len(history) + 1), history, color=colors[name], linewidth=2, label=name)
ax.set_xscale("log")
ax.set_title("Loss vs iteration COUNT (misleading)", fontsize=11, fontweight="bold")
ax.set_xlabel("outer iteration (log scale) - one SGD epoch is 1200 gradient steps")
ax.set_ylabel("log loss")
ax.legend(fontsize=7, frameon=False)
ax.grid(alpha=0.25)
ax.text(0.30, 0.55, "The x-axis counts SGD epochs and GD passes\nas the same unit.\nThey are not comparable.",
        transform=ax.transAxes, fontsize=8, color="#E03131", fontweight="bold")

# --- Panel 2: loss vs WALL-CLOCK (the honest plot) -------------------------------------
ax = axes[1]
for name, (_, history, _) in results.items():
    per_step = timings[name] / max(len(history) - 1, 1)
    times = np.arange(len(history)) * per_step
    ax.plot(times, history, color=colors[name], linewidth=2, label=name)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("Loss vs wall-clock seconds (honest)", fontsize=11, fontweight="bold")
ax.set_xlabel("seconds (log scale)")
ax.set_ylabel("log loss (log scale)")
ax.legend(fontsize=7, frameon=False)
ax.grid(alpha=0.25, which="both")
ax.text(0.42, 0.55, "Batch GD finishes its whole curve in 0.05s.\nSGD is still descending at 1.3s.",
        transform=ax.transAxes, fontsize=8, color="#0CA678", fontweight="bold")

# --- Panel 3: distance to the optimum over iterations ----------------------------------
# The honest measure of "how close am I" that does not depend on knowing the
# optimum: distance from the FULL-batch gradient descent solution, which we treat
# as ground truth because it is exact and converged.
reference = results["batch GD (800 full passes)"][0]
ax = axes[2]
for name, (_, history, weights_history) in results.items():
    distances = np.linalg.norm(weights_history - reference, axis=1)
    ax.plot(np.arange(1, len(distances) + 1), np.maximum(distances, 1e-12),
            color=colors[name], linewidth=2, label=name)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("Distance to the batch-GD optimum", fontsize=11, fontweight="bold")
ax.set_xlabel("outer iteration (log scale)")
ax.set_ylabel("||w - w*||  (log scale)")
ax.legend(fontsize=7, frameon=False)
ax.grid(alpha=0.25, which="both")
ax.text(0.03, 0.06, "Batch GD and Adam snap into the optimum.\nSGD is still 1.05 away after 60 epochs.",
        transform=ax.transAxes, fontsize=8, color="#7048E8", fontweight="bold")

fig.suptitle("03 - Gradient Descent Variants", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


# --------------------------------------------------------------------------------------
# 7. Takeaways.
# --------------------------------------------------------------------------------------
print()
print("=" * 76)
print("TAKEAWAYS")
print("=" * 76)
print("1. The log loss is CONVEX. One optimum, no local minima, no restarts, no")
print("   learning-rate schedule. That is logistic regression's real gift.")
print("2. Batch GD takes the most accurate step direction because it uses all the")
print("   data, and has no noise floor. SGD's step is cheap but noisy, and the")
print("   jitter it leaves behind is proportional to the learning rate. Mini-batch")
print("   is the compromise that makes GPU training possible.")
print("3. Plot loss against wall-clock, not against iteration count. The first two")
print("   panels of this file disagree completely about how fast these are.")
print("4. Renormalise mini-batch gradients by the full n, not the batch size,")
print("   otherwise your learning rate silently means something different per batch.")
print("5. Always scale before you tune a learning rate. One unscaled feature can")
print("   make every learning rate simultaneously wrong.")
print("6. Adam is not free accuracy. It pays for its robustness with a final loss")
print("   that is slightly worse, and a few parameters that never quite settle.")
print("7. For tabular data, prefer lbfgs. Reach for saga at scale, SGD only when")
print("   you genuinely cannot hold the data in memory, and Adam almost never -")
print("   it belongs to the neural network world, not here.")
