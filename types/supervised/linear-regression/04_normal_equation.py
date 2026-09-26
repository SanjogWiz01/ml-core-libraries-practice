"""
04 - The Normal Equation (Closed-Form Least Squares)
=====================================================

Goal: solve the linear regression problem in one exact shot with no learning rate
and no iterations, and understand precisely when this stops being a good idea.

The derivation
--------------
We want to minimise MSE = (1/n)||y - Xw||^2. Set its gradient to zero:

    d/dw (y - Xw)^T (y - Xw) = -2 X^T (y - Xw) = 0
    =>  X^T X w = X^T y
    =>  w = (X^T X)^(-1) X^T y

That is the normal equation. The exact minimiser of MSE, in one multiplication.

Requirements and failure modes
-------------------------------
1. `X^T X` must be invertible -> features must be linearly independent.
   Perfectly collinear columns make it singular. We use `pinv` (SVD-based) so the
   code still returns something, but the coefficients are then non-unique.
2. `X^T X` squares the condition number. If `X^T X` has condition number kappa,
   the solution's error is inflated roughly by kappa. Doubly bad news for
   normalisation-sensitive pipelines.
3. Computing the inverse is wasteful - the mathematically stable move is to solve
   the system, or use an SVD. scikit-learn's `LinearRegression` does exactly that
   via `scipy.linalg.lstsq` and never forms the inverse.

Run:  python 04_normal_equation.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes

SEED = 42
rng = np.random.default_rng(SEED)

print("=" * 76)
print("THE NORMAL EQUATION - exact linear regression in one line of maths")
print("=" * 76)


# ======================================================================================
# PART 1 - build it by hand and verify it against scikit-learn
# ======================================================================================
print()
print("-" * 76)
print("PART 1 - hand-computed vs scikit-learn")
print("-" * 76)

n = 250
X = rng.normal(0, 1, size=(n, 3))
true_w = np.array([2.5, -1.2, 0.7])
y = X @ true_w + 4.0 + rng.normal(0, 0.5, size=n)

X_bias = np.hstack([np.ones((n, 1)), X])

# --- Method A: the textbook inverse --------------------------------------------------
XTX_inv = np.linalg.inv(X_bias.T @ X_bias)
w_inverse = XTX_inv @ (X_bias.T @ y)

# --- Method B: the pseudo-inverse, which is what you should actually type ----------
# `pinv` is computed with SVD, so it degrades gracefully on singular matrices and
# is numerically better behaved than `inv`. Prefer it, or use lstsq directly.
w_pinv = np.linalg.pinv(X_bias.T @ X_bias) @ (X_bias.T @ y)

# --- Method C: solve the linear SYSTEM without ever forming the inverse -------------
# This is the most correct option: it avoids the explicit inverse entirely.
w_solve, *_ = np.linalg.lstsq(X_bias, y, rcond=None)

# --- Method D: let scikit-learn do it -----------------------------------------------
sk_model = LinearRegression().fit(X, y)
w_sklearn = np.concatenate([[sk_model.intercept_], sk_model.coef_])

comparison = pd.DataFrame({
    "intercept": [w[0] for w in (w_inverse, w_pinv, w_solve, w_sklearn)],
    "w1": [w[1] for w in (w_inverse, w_pinv, w_solve, w_sklearn)],
    "w2": [w[2] for w in (w_inverse, w_pinv, w_solve, w_sklearn)],
    "w3": [w[3] for w in (w_inverse, w_pinv, w_solve, w_sklearn)],
}, index=["inv(XtX)XtY", "pinv(XtX)XtY", "lstsq (no inverse)", "sklearn LinearRegression"])

print("Truth: intercept 4.0000, w = [2.5, -1.2, 0.7]")
print()
print(comparison.round(8).to_string())
print()
print("All four agree to ~8 decimal places. Notice the machine-precision noise")
print("between them - that is float64 arithmetic, not a real difference.")

# MSE comparison proves they are equivalent as models, not just similar numbers.
print()
for name, w in [("inv", w_inverse), ("pinv", w_pinv), ("lstsq", w_solve), ("sklearn", w_sklearn)]:
    mse = np.mean((y - X_bias @ w) ** 2)
    print(f"  MSE via {name:<9} = {mse:.12f}")
print("=> identical. Pick lstsq/pinv and stop worrying about this.")


# ======================================================================================
# PART 2 - singularity: what happens when a feature is a duplicate
# ======================================================================================
print()
print("-" * 76)
print("PART 2 - the singular case: duplicate features")
print("-" * 76)

X_dup = np.hstack([X, X[:, [0]]])  # x1 appears twice, identically
X_dup_bias = np.hstack([np.ones((n, 1)), X_dup])
y_dup = X_dup[:, :3] @ true_w + 4.0 + rng.normal(0, 0.5, size=n)

det = np.linalg.det(X_dup_bias.T @ X_dup_bias)
print(f"det(X^T X) with the duplicate column = {det:.6e}")
print("It is (numerically) zero: x1 and x1_copy are linearly dependent, so the")
print("columns of X^T X are dependent and the matrix has no inverse.")
print()

try:
    w_blowup = np.linalg.inv(X_dup_bias.T @ X_dup_bias) @ (X_dup_bias.T @ y_dup)
    print(f"inv() did not raise, but produced: {w_blowup.round(4)}")
    print("  ^ This is the danger: a singular matrix often does NOT error out, it")
    print("    just returns huge, meaningless numbers.")
except np.linalg.LinAlgError as exc:
    print(f"inv() raised LinAlgError: {exc}")

# pinv and lstsq both return a valid minimum-norm solution. The duplicate column
# gets a coefficient and the original gets 0 (or they share the weight) - the
# model still predicts correctly, but "which feature matters" is unanswerable.
w_pinv_dup = np.linalg.pinv(X_dup_bias.T @ X_dup_bias) @ (X_dup_bias.T @ y_dup)
w_lstsq_dup, *_ = np.linalg.lstsq(X_dup_bias, y_dup, rcond=None)
print()
print("Minimum-norm solutions instead of an inverse:")
print(f"  pinv  : {w_pinv_dup.round(6)}")
print(f"  lstsq : {w_lstsq_dup.round(6)}")
print(f"  sum of the two x1 columns: {w_pinv_dup[1] + w_pinv_dup[4]:.6f}  (true coefficient 2.5)")
print()
print("PREDICTIONS are still right; individual COEFFICIENTS are meaningless.")
print("That is the whole lesson of collinearity: fine as a predictor, useless as")
print("an explanation. It is also why ridge exists - see script 12.")


# ======================================================================================
# PART 3 - conditioning: badly scaled features wreck the explicit inverse
# ======================================================================================
print()
print("-" * 76)
print("PART 3 - condition number: why you should never type inv()")
print("-" * 76)

# Build a matrix whose columns are nearly (but not exactly) dependent.
eps = 1e-8
X_ill = np.column_stack([
    X[:, 0],
    X[:, 0] + eps,       # near-duplicate of column 0
    X[:, 1] * 1e6,       # huge scale
    X[:, 2] * 1e-6,      # tiny scale
])
X_ill_bias = np.hstack([np.ones((n, 1)), X_ill])
y_ill = X_ill @ np.array([2.0, 1.0, -0.5, 0.3]) + 4.0

# Condition number of X itself, and of the Gram matrix X^T X.
cond_X = np.linalg.cond(X_ill)
cond_XtX = np.linalg.cond(X_ill_bias.T @ X_ill_bias)
print(f"cond(X)      = {cond_X:.4e}")
print(f"cond(X^T X)  = {cond_XtX:.4e}    <- roughly cond(X)^2 = {cond_X ** 2:.4e}")
print()
print("The Gram matrix squares the conditioning problem. Any noise in the data is")
print("amplified twice on the way into the weights.")
print()

# Ridge with a tiny penalty is the numerically well-conditioned route to the
# same answer. Note how much closer it stays to a stable reference solution.
sc = StandardScaler()
X_scaled = sc.fit_transform(X_ill)
X_scaled_bias = np.hstack([np.ones((n, 1)), X_scaled])

w_stable = np.linalg.pinv(X_scaled_bias.T @ X_scaled_bias) @ (X_scaled_bias.T @ y_ill)
w_ridge = Ridge(alpha=1e-8, fit_intercept=False).fit(X_ill, y_ill)
ridge_coef = np.concatenate([[w_ridge.intercept_], w_ridge.coef_])

print(f"max |coef| from pinv on raw ill-conditioned X : {np.max(np.abs(np.linalg.pinv(X_ill_bias.T @ X_ill_bias) @ (X_ill_bias.T @ y_ill))):.4e}")
print(f"max |coef| from pinv on STANDARDISED X        : {np.max(np.abs(w_stable)):.4e}")
print(f"max |coef| from Ridge(alpha=1e-8), unstandard : {np.max(np.abs(ridge_coef)):.4e}")
print()
print("Standardising first, or adding a small ridge penalty, tames the explosion.")
print("These are two views of the same fix: keep the geometry of the problem sane.")


# ======================================================================================
# PART 4 - realistic case, and why scikit-learn's default is the right default
# ======================================================================================
print()
print("-" * 76)
print("PART 4 - on real data, with a timing comparison")
print("-" * 76)

diabetes = load_diabetes()
X_real = diabetes.data
y_real = diabetes.target
X_real_bias = np.hstack([np.ones((X_real.shape[0], 1)), X_real])

import time  # noqa: E402

start = time.perf_counter()
_ = np.linalg.pinv(X_real_bias.T @ X_real_bias) @ (X_real_bias.T @ y_real)
time_pinv = time.perf_counter() - start

start = time.perf_counter()
_ = LinearRegression().fit(X_real, y_real)
time_sklearn = time.perf_counter() - start

print(f"diabetes dataset: {X_real.shape[0]} samples, {X_real.shape[1]} features")
print(f"  explicit pinv(X^T X) X^T y : {time_pinv * 1000:8.3f} ms   (builds an {X_real.shape[1]+1}x{X_real.shape[1]+1} inverse)")
print(f"  sklearn LinearRegression   : {time_sklearn * 1000:8.3f} ms   (uses LAPACK lstsq / SVD)")
print()
print("For p features the normal equation costs O(p^3). At p = 10,000 that is")
print("10^12 operations and about 800 MB just for X^T X. Gradient descent instead")
print("costs O(n*p) per step and never materialises that matrix. This is the")
print("practical crossover: closed form while p is small, iterative once it is not.")


# ======================================================================================
# Summary
# ======================================================================================
print()
print("=" * 76)
print("SUMMARY")
print("=" * 76)
print("1. w = (X^T X)^-1 X^T y is the EXACT minimiser of MSE. No iterations needed.")
print("2. Prefer np.linalg.lstsq or pinv over inv() - never form the inverse explicitly.")
print("3. Singular X^T X (collinear features) breaks the interpretation of the")
print("   coefficients, though predictions can still be fine.")
print("4. The Gram matrix squares the condition number, so scaling or a small ridge")
print("   penalty makes the whole problem numerically kinder.")
print("5. Use the closed form when p is small; use SGD/Ridge when p is large or you")
print("   need a penalty. That is exactly what LinearRegression vs SGDRegressor is.")
