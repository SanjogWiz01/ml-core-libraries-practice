"""
03 - Gradient Descent Variants
==============================

Goal: understand the four optimisation strategies you will meet in practice, by
implementing and timing all of them yourself.

The four
--------
1. **Batch GD** - use all n samples for every update. Smooth, guaranteed-ish
   convergence, but one update per epoch is slow on big data.
2. **Stochastic GD (SGD)** - one random sample per update. Very fast per epoch,
   extremely noisy loss, needs many epochs and a decaying learning rate.
3. **Mini-batch GD** - a small batch (32-256 typical). This is what almost every
   real implementation does: the noise helps escape sharp minima and the compute
   stays cache-friendly.
4. **Adaptive (Adam / RMSProp)** - per-parameter learning rates estimated from
   running gradient moments. Fast convergence, less tuning.

Key tradeoffs measured in this file
-----------------------------------
- Updates needed to converge
- Wall-clock seconds
- Quality of the final answer (all should land on the same optimum)

Run:  python 03_gradient_descent_variants.py
"""

import time

import matplotlib.pyplot as plt
import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)

print("=" * 78)
print("GRADIENT DESCENT VARIANTS - all four implemented from scratch")
print("=" * 78)


# ======================================================================================
# Data: 4 features on very different scales. The scale difference is deliberate -
# it is exactly the situation that makes plain batch GD crawl.
# ======================================================================================
n_samples, n_features = 400, 4
true_coef = np.array([3.0, -1.5, 0.8, 0.0])
true_intercept = 5.0

X_raw = rng.normal(0, 1, size=(n_samples, n_features))
X_raw[:, 0] *= 1000.0  # feature 0 lives in the thousands
X_raw[:, 1] *= 0.01   # feature 1 lives in the hundredths
X_raw[:, 2] *= 1.0
X_raw[:, 3] *= 0.1

y = X_raw @ true_coef + true_intercept + rng.normal(0, 1.0, size=n_samples)
X_bias = np.hstack([np.ones((n_samples, 1)), X_raw])

print(f"\nFeature standard deviations: {X_raw.std(axis=0).round(4)}")
print("  <- a 100,000x spread. This single fact explains most of the differences below.")
print(f"\nHidden truth: intercept {true_intercept}, coefficients {true_coef}")


# ======================================================================================
# The four optimisers. All share the same loss and the same gradient function, so
# any difference in the result is down to the update strategy alone.
# ======================================================================================
def mse_and_gradient(X, y, w):
    """Return (MSE, gradient of MSE) for the design matrix X (first column = ones)."""
    n = X.shape[0]
    residual = y - X @ w
    loss = np.mean(residual**2)
    gradient = -(2.0 / n) * (X.T @ residual)
    return loss, gradient


def run_batch_gd(X, y, lr=0.0002, n_iterations=6000):
    """One update per full pass over the data. Deterministic and smooth."""
    w = np.zeros(X.shape[1])
    history = []
    for _ in range(n_iterations):
        loss, grad = mse_and_gradient(X, y, w)
        w -= lr * grad
        history.append(loss)
    return w, history


def run_sgd(X, y, lr=0.0002, n_epochs=60):
    """
    One sample per update. Fast progress, very noisy.
    The learning rate decays each epoch - without this, SGD bounces around forever.
    """
    w = np.zeros(X.shape[1])
    n = X.shape[0]
    history = []
    for epoch in range(n_epochs):
        current_lr = lr / (1 + 0.05 * epoch)  # 1/(1+decay*epoch) schedule
        for i in rng.permutation(n):
            x_i = X[i].reshape(1, -1)
            residual = (y[i] - x_i @ w).item()
            grad = -2 * x_i.ravel() * residual
            w -= current_lr * grad
        history.append(mse_and_gradient(X, y, w)[0])
    return w, history


def run_mini_batch_gd(X, y, lr=0.0002, n_epochs=200, batch_size=32):
    """
    A small batch per update. The workhorse: noisy enough to generalise, big enough
    to be fast in vectorised code.
    """
    w = np.zeros(X.shape[1])
    n = X.shape[0]
    history = []
    for _ in range(n_epochs):
        indices = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = indices[start:start + batch_size]
            loss, grad = mse_and_gradient(X[batch], y[batch], w)
            w -= lr * grad * len(batch) / batch_size  # correct for the partial batch
        history.append(loss)
    return w, history


def run_adam(X, y, lr=0.05, n_iterations=6000):
    """
    Adam: momentum (first moment) + adaptive scaling (second moment).
    Each parameter gets its own effective step size, which is exactly what makes
    badly scaled features tractable.
    """
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    w = np.zeros(X.shape[1])
    m = np.zeros_like(w)  # running mean of gradients
    v = np.zeros_like(w)  # running mean of squared gradients
    history = []
    for t in range(1, n_iterations + 1):
        loss, grad = mse_and_gradient(X, y, w)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)  # bias correction, important for the first steps
        v_hat = v / (1 - beta2**t)
        w -= lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(loss)
    return w, history


# ======================================================================================
# Run all four and measure them
# ======================================================================================
print()
print("-" * 78)
print("Running all four optimisers (identical loss, identical gradient)")
print("-" * 78)
print(f"{'variant':<22} {'seconds':>9} {'final MSE':>12} {'max coef error':>16}")
print("-" * 78)

results = {}
timings = {}

for name, fn in [
    ("batch GD", run_batch_gd),
    ("stochastic GD", run_sgd),
    ("mini-batch GD", run_mini_batch_gd),
    ("Adam", run_adam),
]:
    start = time.perf_counter()
    weights, history = fn(X_bias, y)
    elapsed = time.perf_counter() - start
    # How far are we from the true coefficients? (The optimum is not exactly the
    # truth because of noise, so we also compare against the closed-form solution.)
    error = np.max(np.abs(weights[1:] - true_coef))
    results[name] = (weights, np.array(history))
    timings[name] = elapsed
    print(f"{name:<22} {elapsed:>8.3f}s {history[-1]:>12.5f} {error:>16.4f}")

optimal = np.linalg.pinv(X_bias.T @ X_bias) @ (X_bias.T @ y)
print("-" * 78)
print(f"{'normal equation':<22} {'~0s':>9} {np.mean((y - X_bias @ optimal) ** 2):>12.5f} "
      f"{np.max(np.abs(optimal[1:] - true_coef)):>16.4f}")
print()
print("Every optimiser reaches the same loss. The difference is HOW MUCH WORK it")
print("took to get there, not WHERE it ended up. That is the whole point of this file.")


# ======================================================================================
# Now repeat on standardised data. The gap between batch GD and Adam mostly vanishes.
# ======================================================================================
print()
print("-" * 78)
print("Repeating on STANDARDISED features (mean 0, std 1)")
print("-" * 78)

means = X_raw.mean(axis=0)
stds = X_raw.std(axis=0)
X_std = (X_raw - means) / stds
X_bias_std = np.hstack([np.ones((n_samples, 1)), X_std])
y_std = y - y.mean()  # centre the target too, otherwise the intercept does all the work

print(f"{'variant':<22} {'seconds':>9} {'final MSE (std space)':>24}")
print("-" * 78)
for name, fn in [
    ("batch GD", run_batch_gd),
    ("stochastic GD", run_sgd),
    ("mini-batch GD", run_mini_batch_gd),
    ("Adam", run_adam),
]:
    start = time.perf_counter()
    weights, history = fn(X_bias_std, y_std, lr=0.01)  # a much larger lr is now safe
    elapsed = time.perf_counter() - start
    results[name + " (std)"] = (weights, np.array(history))
    print(f"{name:<22} {elapsed:>8.3f}s {history[-1]:>24.5f}")

print()
print("Once features share a scale, one learning rate works for all of them. This")
print("is the single most important practical reason to call StandardScaler.")

# Undo the centring so the standardised coefficients are comparable to the raw ones.
print()
print("Coefficient comparison (note the 1000x scale difference on feature 0):")
print(f"{'feature':<12} {'std of feature':>15} {'coef (raw data)':>16} {'coef (std data)':>17}")
print("-" * 78)
for j in range(n_features):
    raw_coef = results["batch GD"][0][j + 1]
    std_coef = results["batch GD (std)"][0][j + 1]
    print(f"x{j:<11} {stds[j]:>15.4f} {raw_coef:>16.4f} {std_coef:>17.4f}")
print()
print("The raw coefficient looks tiny and unimportant. The standardised one is")
print("comparable to the others. The raw number was measuring units, not effect.")


# ======================================================================================
# Visual comparison of the convergence paths
# ======================================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))

# --- Panel 1: raw features, all four optimisers ---------------------------------------
ax = axes[0]
colors = {
    "batch GD": "#4C6EF5",
    "stochastic GD": "#F59F00",
    "mini-batch GD": "#0CA678",
    "Adam": "#E03131",
}
for name, color in colors.items():
    history = results[name][1]
    ax.plot(np.arange(1, len(history) + 1), history, color=color, linewidth=1.5,
            alpha=0.9, label=f"{name} ({timings[name]:.2f}s)")
ax.set_yscale("log")
ax.set_title("Raw features (100,000x scale spread)", fontsize=11, fontweight="bold")
ax.set_xlabel("iterations (or epochs for SGD/mini-batch)")
ax.set_ylabel("MSE (log scale)")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Panel 2: standardised features ---------------------------------------------------
ax = axes[1]
for name, color in colors.items():
    history = results[name + " (std)"][1]
    ax.plot(np.arange(1, len(history) + 1), history, color=color, linewidth=1.5,
            alpha=0.9, label=name)
ax.set_yscale("log")
ax.set_title("Standardised features - all four converge quickly", fontsize=11,
             fontweight="bold")
ax.set_xlabel("iterations (or epochs)")
ax.set_ylabel("MSE (log scale)")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

fig.suptitle("03 - Gradient Descent Variants", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


# ======================================================================================
# The same four, using scikit-learn's SGDRegressor
# ======================================================================================
print()
print("-" * 78)
print("scikit-learn equivalent (what you would actually write at work)")
print("-" * 78)

from sklearn.linear_model import SGDRegressor  # noqa: E402  (import placed here on purpose)

for loss in ["squared_error", "huber", "epsilon_insensitive"]:
    sgd = SGDRegressor(
        loss=loss,
        penalty="l2",
        alpha=1e-4,
        learning_rate="optimal",   # scikit-learn's schedule
        max_iter=1000,
        tol=1e-4,
        random_state=SEED,
    )
    sgd.fit(X_std, y_std)
    score = sgd.score(X_std, y_std)
    print(f"  loss={loss:<20} R2 on train = {score:.4f}   intercept_ = {sgd.intercept_[0]:.3f}")

print()
print("For plain OLS you usually do NOT need gradient descent at all:")
print("  LinearRegression -> closed-form least squares (exact, instant)")
print("  SGDRegressor     -> needed only for huge n or p, or to add penalties")
print("  Ridge/SGDRegressor with penalty -> when features are many and correlated")

print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
print("1. Batch GD: smooth and stable, one update per epoch, too slow for big n.")
print("2. SGD: fastest per epoch, very noisy, needs a decaying learning rate.")
print("3. Mini-batch: what everyone uses. Noise helps, vectorisation helps more.")
print("4. Adam: adaptive per-feature step sizes, converges fastest, no tuning needed.")
print("5. All four solve the SAME problem. Badly scaled features are why some of")
print("   them struggle - fix the scaling and the differences mostly disappear.")
print("6. If you have no reason to iterate, use the closed form (script 04).")
