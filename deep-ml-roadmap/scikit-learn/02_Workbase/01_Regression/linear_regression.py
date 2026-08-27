"""Linear Regression — fundamentals of regression in scikit-learn."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def load_data():
    housing = fetch_california_housing(as_frame=True)
    X, y = housing.data, housing.target  # target is median house value in $100k
    return X, y, housing.feature_names


def evaluate(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"{label}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    return mae, rmse, r2


def plot_predictions(y_true, y_pred, title="Predicted vs Actual"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(y_true, y_pred, alpha=0.3, s=10)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title(title)
    axes[0].legend()

    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[1].axhline(0, color='r', linestyle='--')
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residual Plot")

    plt.tight_layout()
    plt.show()


def main():
    print("=" * 55)
    print("LINEAR REGRESSION — California Housing Dataset")
    print("=" * 55)

    X, y, feature_names = load_data()
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}] ($100k units)")

    # --- 1. Split first, always ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- 2. Scale (fit on train, transform both) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # --- 3. Train ---
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # --- 4. Evaluate ---
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test  = model.predict(X_test_scaled)

    print("\n--- Performance ---")
    evaluate(y_train, y_pred_train, "Train set:")
    evaluate(y_test,  y_pred_test,  "Test set: ")

    # --- 5. Coefficients ---
    print("\n--- Feature Coefficients (scaled) ---")
    coeff_pairs = sorted(
        zip(feature_names, model.coef_),
        key=lambda x: abs(x[1]), reverse=True
    )
    for name, coef in coeff_pairs:
        print(f"  {name:25s}: {coef:+.4f}")

    print(f"\n  Intercept: {model.intercept_:.4f}")

    # --- 6. Visualize ---
    plot_predictions(y_test.values, y_pred_test)


if __name__ == "__main__":
    main()
