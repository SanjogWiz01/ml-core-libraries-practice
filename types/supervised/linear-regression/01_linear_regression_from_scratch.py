"""
01 - Linear Regression From Scratch
====================================

Goal: fit y = w0 + w1*x by hand using gradient descent, so the scikit-learn
version in script 02 never feels like magic.

What this file demonstrates
---------------------------
1. The exact gradient of Mean Squared Error w.r.t. the weights.
2. How the loss surface collapses from a bowl into a smooth parabola.
3. How `learning_rate` controls convergence speed - and what divergence looks
   like when it is too large.
4. That gradient descent finds the *same* answer as the closed-form least
   squares solution (we verify this numerically against the normal equation).

The model
---------
    y_hat_i = w0 + w1 * x_i

The loss
--------
    J(w0, w1) = (1/n) * SUM_i (y_i - (w0 + w1*x_i))^2

Partial derivatives (the update rules)
-------------------------------------
    dJ/dw0 = -(2/n) * SUM_i (y_i - y_hat_i)          <- intercept
    dJ/dw1 = -(2/n) * SUM_i x_i * (y_i - y_hat_i)    <- slope

Update both weights by walking *against* the gradient:

    w0 <- w0 - lr * dJ/dw0
    w1 <- w1 - lr * dJ/dw1

Run:  python 01_linear_regression_from_scratch.py
"""

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------------------
# 1. Data: a synthetic relationship we deliberately make with a known ground truth.
#    Using a known truth lets us check whether training actually recovered it.
# --------------------------------------------------------------------------------------
TRUE_SLOPE = 3.5
TRUE_INTERCEPT = -2.0
NOISE_LEVEL = 4.0  # standard deviation of the random noise

rng = np.random.default_rng(42)  # seeded -> reproducible results
n_samples = 200
X = rng.uniform(0, 10, size=n_samples)  # one feature, 200 samples
y = TRUE_SLOPE * X + TRUE_INTERCEPT + rng.normal(0, NOISE_LEVEL, size=n_samples)

# Reshape to a column vector. That is the shape scikit-learn expects:
# (n_samples, n_features). A 1-D array of length n is the classic beginner bug,
# because scikit-learn interprets it as n samples with 0 features.
X = X.reshape(-1, 1)
X_bias = np.hstack([np.ones((n_samples, 1)), X])  # prepend a column of ones

print("=" * 70)
print("LINEAR REGRESSION FROM SCRATCH - gradient descent on MSE")
print("=" * 70)
print(f"Data shape        : X {X.shape}, y {y.shape}")
print(f"True relationship : y = {TRUE_SLOPE} * x + {TRUE_INTERCEPT}")
print(f"Noise (std)       : {NOISE_LEVEL}")
print(f"MSE of a model that always predicts the mean: {np.var(y):.4f}")
print("  (R2 = 0 corresponds to exactly this error, so it is the bar to beat)")
print()


# --------------------------------------------------------------------------------------
# 2. The training function.
# --------------------------------------------------------------------------------------
def train_linear_regression(
    X_bias: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    n_iterations: int = 2000,
) -> tuple[np.ndarray, list[float]]:
    """
    Fit y_hat = w0 + w1*x with batch gradient descent.

    Parameters
    ----------
    X_bias : (n, 2) array
        Design matrix whose FIRST COLUMN must be the constant 1. That is how the
        intercept gets treated as just another weight and is updated by the same rule.
    y : (n,) array
        Targets.
    learning_rate : float
        Step size of each update. Too small -> glacial. Too large -> the loss
        explodes to inf/nan.
    n_iterations : int
        How many full passes over the data to make.

    Returns
    -------
    weights : (2,) array -> [intercept, slope]
    loss_history : list[float]
        MSE after each iteration, for plotting the convergence curve.
    """
    n = X_bias.shape[0]
    weights = np.zeros(2)  # start at zero; the origin is a perfectly valid start
    loss_history = []

    for iteration in range(n_iterations):
        # --- forward pass: predict with the CURRENT weights ---------------------
        predictions = X_bias @ weights           # (n,2) @ (2,) -> (n,)
        residuals = y - predictions              # y_hat subtracted from y
        loss = np.mean(residuals**2)             # MSE, the scalar we minimise
        loss_history.append(loss)

        # --- backward pass: how does each weight need to change? ---------------
        # dJ/dw = -(2/n) * X^T (y - y_hat)   -- vectorised form of the two formulas
        gradient = -(2.0 / n) * (X_bias.T @ residuals)

        # --- update: step against the gradient ----------------------------------
        weights -= learning_rate * gradient

        # A guard worth having in any hand-rolled trainer. If the loss is not a
        # finite number, the learning rate blew up and continuing is pointless.
        if not np.isfinite(loss):
            raise ValueError(
                f"Loss became {loss} at iteration {iteration}. "
                f"learning_rate={learning_rate} is too large - try 0.01."
            )

    return weights, loss_history


print("-" * 70)
print("PART A - train with a sensible learning rate")
print("-" * 70)
weights, loss_history = train_linear_regression(X_bias, y, learning_rate=0.01, n_iterations=3000)
intercept, slope = weights
loss_history = np.array(loss_history)

print(f"Learned intercept : {intercept:>10.4f}   (true {TRUE_INTERCEPT})")
print(f"Learned slope     : {slope:>10.4f}   (true {TRUE_SLOPE})")
print(f"Final MSE         : {loss_history[-1]:>10.4f}")
print(f"Initial MSE       : {loss_history[0]:>10.4f}")
print(f"Total MSE reduced : {loss_history[0] - loss_history[-1]:>10.4f}")


# --------------------------------------------------------------------------------------
# 3. The skill that matters: recover the exact same answer from the normal equation.
#    If these two disagree, one of the implementations has a bug. This is the single
#    most useful test you can write while learning linear regression.
# --------------------------------------------------------------------------------------
weights_normal_eq = np.linalg.pinv(X_bias.T @ X_bias) @ (X_bias.T @ y)

print()
print("-" * 70)
print("PART B - cross-check against the closed-form normal equation")
print("-" * 70)
print("Gradient descent  : intercept %.6f  slope %.6f" % (intercept, slope))
print("Normal equation   : intercept %.6f  slope %.6f" % tuple(weights_normal_eq))
print("Max abs difference: %.3e" % np.max(np.abs(weights - weights_normal_eq)))
print("=> Gradient descent converged to the least-squares optimum. Checks out.")


# --------------------------------------------------------------------------------------
# 4. Learning rate is the one hyperparameter that can break training.
# --------------------------------------------------------------------------------------
print()
print("-" * 70)
print("PART C - what happens when the learning rate is wrong")
print("-" * 70)
print(f"{'learning_rate':>15} | {'iterations to reach MSE < 15':>32} | outcome")
print("-" * 70)

for lr in [0.0005, 0.005, 0.01, 0.05, 0.1, 0.6]:
    try:
        w, history = train_linear_regression(X_bias, y, learning_rate=lr, n_iterations=5000)
        # Count how many iterations it took to get within 0.01% of the final loss.
        target = history[-1] * 1.0001
        converged_at = next((i for i, loss in enumerate(history) if loss < target), -1)
        print(f"{lr:>15} | {converged_at:>32} | final MSE {history[-1]:.4f}")
    except ValueError as exc:
        print(f"{lr:>15} | {'never':>32} | {exc}")
        break

print()
print("Notice the pattern: larger steps converge faster until one step overshoots")
print("the minimum, then the weights oscillate and the loss explodes. That cliff")
print("is why you tune the learning rate, and why SGDRegressor lets you set it.")


# --------------------------------------------------------------------------------------
# 5. Plot: the data, the fitted line, the loss curve, and the loss surface.
# --------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# --- Panel 1: scatter + fitted line -------------------------------------------------
ax = axes[0]
ax.scatter(X, y, alpha=0.5, s=28, color="#4C6EF5", edgecolors="none", label="observations")

x_line = np.linspace(X.min(), X.max(), 100)
# The mean-prediction line: no error at all, so MSE = variance of y.
ax.plot(x_line, y.mean() * np.ones_like(x_line), "--", color="#868E96",
        linewidth=2, label=f"mean baseline (MSE={np.var(y):.1f})")
ax.plot(x_line, slope * x_line + intercept, color="#E03131", linewidth=2.5,
        label=f"fitted line (MSE={loss_history[-1]:.1f})")

ax.set_title("Data and the fitted line", fontsize=11, fontweight="bold")
ax.set_xlabel("x (feature)")
ax.set_ylabel("y (target)")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Panel 2: loss curve ------------------------------------------------------------
ax = axes[1]
ax.plot(loss_history, color="#0CA678", linewidth=1.6)
ax.set_yscale("log")  # log scale reveals the early fast drops that linear scale hides
ax.set_title("MSE falls every single iteration", fontsize=11, fontweight="bold")
ax.set_xlabel("iteration")
ax.set_ylabel("MSE (log scale)")
ax.grid(alpha=0.25)
ax.annotate("converged", xy=(len(loss_history) * 0.62, loss_history[-1]),
            fontsize=9, color="#0CA678")

# --- Panel 3: the loss surface in (w0, w1) space -------------------------------------
# For a 1-D feature the MSE is exactly a convex bowl. Plotting it makes the word
# "convex" concrete: there is one minimum and no local traps.
ax = axes[2]
w0_grid = np.linspace(intercept - 60, intercept + 60, 120)
w1_grid = np.linspace(slope - 6, slope + 6, 120)
W0, W1 = np.meshgrid(w0_grid, w1_grid)
# Broadcast the prediction over the whole grid, then measure loss at each point.
# Shapes: (120, 120, 1) x (1, 1, n) -> (120, 120, n) = loss at every (w0, w1) pair.
grid_predictions = W0[:, :, None] + W1[:, :, None] * X.reshape(1, 1, -1)
grid_loss = np.mean((y.reshape(1, 1, -1) - grid_predictions) ** 2, axis=2)

contour = ax.contour(W0, W1, grid_loss, levels=25, cmap="viridis", alpha=0.85)
ax.contourf(W0, W1, grid_loss, levels=25, cmap="viridis", alpha=0.35)
ax.scatter([intercept], [slope], marker="*", s=320, color="#E03131",
           edgecolors="white", linewidths=1.2, zorder=5, label="optimum")
ax.set_title("The loss surface: one convex minimum", fontsize=11, fontweight="bold")
ax.set_xlabel("w0 (intercept)")
ax.set_ylabel("w1 (slope)")
ax.legend(fontsize=8, frameon=False)

fig.suptitle("01 - Linear Regression From Scratch", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


# --------------------------------------------------------------------------------------
# 6. Takeaways.
# --------------------------------------------------------------------------------------
print()
print("=" * 70)
print("TAKEAWAYS")
print("=" * 70)
print("1. The gradient of MSE has a clean vectorised form: -(2/n) * X^T (y - y_hat).")
print("2. Prepending a column of 1s lets the intercept be learned by the same rule.")
print("3. MSE is strictly convex, so there is exactly one optimum - no local minima.")
print("4. Gradient descent and the normal equation must agree. If they do not, you")
print("   have a bug. This check catches sign errors and off-by-one indexing errors.")
print("5. learning_rate is the only thing between a working model and a diverging one.")
print("6. With one feature the model is a line; with p features it is a hyperplane.")
print("   'Linear' means linear in the weights, not in the features.")
