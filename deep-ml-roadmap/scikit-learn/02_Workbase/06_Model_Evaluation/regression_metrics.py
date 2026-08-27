"""Regression metrics — MAE, MSE, RMSE, R² with visual interpretation."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def compute_all_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R²': r2}


def explain_metrics(y_true, y_pred):
    """Print metrics with human-readable interpretation."""
    m = compute_all_metrics(y_true, y_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_true, np.full_like(y_true, y_true.mean())))

    print(f"\n  MAE:  {m['MAE']:.4f}   → avg absolute error is {m['MAE']:.4f} units")
    print(f"  MSE:  {m['MSE']:.4f}   → avg squared error (penalizes outlier errors more)")
    print(f"  RMSE: {m['RMSE']:.4f}  → same units as target, MSE in interpretable form")
    print(f"  R²:   {m['R²']:.4f}   → model explains {m['R²']*100:.1f}% of variance in y")
    print(f"         (Baseline RMSE = {baseline_rmse:.4f} — 'always predict mean')")


def plot_metrics(y_test, y_pred, title):
    """Scatter plot of predicted vs actual + residuals."""
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Actual vs predicted
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    axes[0].scatter(y_test, y_pred, alpha=0.2, s=8)
    axes[0].plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect')
    axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
    axes[0].set_title(f"Actual vs Predicted — {title}")
    axes[0].legend()

    # Residuals vs predicted
    axes[1].scatter(y_pred, residuals, alpha=0.2, s=8)
    axes[1].axhline(0, color='r', linewidth=2, linestyle='--')
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residual Plot")

    # Residual histogram
    axes[2].hist(residuals, bins=40, edgecolor='white')
    axes[2].axvline(0, color='r', linewidth=2)
    axes[2].set_xlabel("Residual"); axes[2].set_ylabel("Count")
    axes[2].set_title("Residual Distribution")

    plt.tight_layout()
    plt.show()


def compare_models(X_train, X_test, y_train, y_test):
    """Compare multiple models on all four metrics."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    models = {
        'Baseline (mean)':     None,
        'LinearRegression':    LinearRegression(),
        'Ridge(α=1.0)':        Ridge(alpha=1.0),
        'RandomForest':        RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting':    GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    print("\n--- Model Comparison ---")
    print(f"  {'Model':25s}  {'MAE':>8}  {'RMSE':>8}  {'R²':>8}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*8}")

    for name, model in models.items():
        if model is None:
            y_pred = np.full_like(y_test, y_train.mean())
        else:
            model.fit(X_tr_s, y_train)
            y_pred = model.predict(X_te_s)

        m = compute_all_metrics(y_test, y_pred)
        print(f"  {name:25s}  {m['MAE']:8.4f}  {m['RMSE']:8.4f}  {m['R²']:8.4f}")

    return models


def main():
    print("=" * 55)
    print("REGRESSION METRICS — MAE, MSE, RMSE, R²")
    print("=" * 55)

    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n--- Metric Explanation (with RandomForest) ---")
    scaler = StandardScaler()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(scaler.fit_transform(X_train), y_train)
    y_pred = rf.predict(scaler.transform(X_test))
    explain_metrics(y_test, y_pred)

    compare_models(X_train, X_test, y_train, y_test)

    # --- When to use each metric ---
    print("\n--- When to use each metric ---")
    print("  MAE:  interpretable, robust to outliers. Use when outliers shouldn't dominate.")
    print("  RMSE: penalizes large errors. Use when big mistakes are especially costly.")
    print("  R²:   relative goodness of fit. Use for comparing models or reporting to stakeholders.")
    print("  MSE:  mainly for loss functions (differentiable). Use internally, not for reporting.")

    plot_metrics(y_test, y_pred, "RandomForest")


if __name__ == "__main__":
    main()
