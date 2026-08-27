"""Ridge, Lasso, ElasticNet — L1/L2 regularization comparison."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")


def score_model(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2  = r2_score(y_test,  model.predict(X_test))
    n_zeros  = np.sum(np.abs(model.coef_) < 1e-6) if hasattr(model, 'coef_') else 0
    print(f"  {name:25s} | train R²={train_r2:.4f} | test R²={test_r2:.4f} | zeroed coefs={n_zeros}")
    return test_r2, model.coef_.copy()


def main():
    print("=" * 60)
    print("REGULARIZATION — Ridge vs Lasso vs ElasticNet")
    print("=" * 60)

    X, y = fetch_california_housing(as_frame=True, return_X_y=True)
    feature_names = fetch_california_housing().feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # --- 1. Compare models at the same alpha ---
    print("\n--- Model comparison (alpha=1.0) ---")
    models = [
        ("LinearRegression",        LinearRegression()),
        ("Ridge(alpha=1.0)",         Ridge(alpha=1.0)),
        ("Lasso(alpha=0.01)",        Lasso(alpha=0.01, max_iter=5000)),
        ("ElasticNet(a=0.01,l1=0.5)", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)),
    ]

    results = {}
    for name, model in models:
        test_r2, coef = score_model(model, X_train_s, X_test_s, y_train, y_test, name)
        results[name] = coef

    # --- 2. Regularization path for Ridge ---
    print("\n--- Ridge: effect of alpha ---")
    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    ridge_r2s = []
    for a in alphas:
        r = Ridge(alpha=a)
        r.fit(X_train_s, y_train)
        ridge_r2s.append(r2_score(y_test, r.predict(X_test_s)))
    best_alpha = alphas[np.argmax(ridge_r2s)]
    print(f"  Best alpha: {best_alpha} → R²={max(ridge_r2s):.4f}")

    # --- 3. Lasso sparsity ---
    print("\n--- Lasso: sparsity at different alphas ---")
    lasso_alphas = [0.001, 0.01, 0.05, 0.1, 0.5]
    for a in lasso_alphas:
        m = Lasso(alpha=a, max_iter=10000)
        m.fit(X_train_s, y_train)
        n_nonzero = np.sum(np.abs(m.coef_) > 1e-6)
        r2 = r2_score(y_test, m.predict(X_test_s))
        print(f"  alpha={a:.3f} → nonzero coefs={n_nonzero}/{len(feature_names)}, R²={r2:.4f}")

    # --- 4. Coefficient visualization ---
    fig, axes = plt.subplots(1, len(alphas), figsize=(18, 4), sharey=True)
    fig.suptitle("Ridge Coefficient Magnitude vs Alpha")
    for i, a in enumerate(alphas):
        r = Ridge(alpha=a)
        r.fit(X_train_s, y_train)
        axes[i].barh(feature_names, r.coef_)
        axes[i].set_title(f"α={a}")
        axes[i].axvline(0, color='k', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # --- 5. R² vs alpha plot ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(alphas, ridge_r2s, 'bo-', label='Ridge')
    lasso_r2s = []
    for a in lasso_alphas:
        m = Lasso(alpha=a, max_iter=10000)
        m.fit(X_train_s, y_train)
        lasso_r2s.append(r2_score(y_test, m.predict(X_test_s)))
    ax.semilogx(lasso_alphas, lasso_r2s, 'rs-', label='Lasso')
    ax.set_xlabel("Alpha (log scale)")
    ax.set_ylabel("Test R²")
    ax.set_title("Regularization Strength vs Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
