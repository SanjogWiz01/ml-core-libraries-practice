"""Gradient Boosting — sequential ensemble, key hyperparameters."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.metrics import accuracy_score, r2_score
import warnings

warnings.filterwarnings("ignore")


def staged_score(model, X_test, y_test, is_classifier=True):
    """Return accuracy/R² after each boosting stage (for learning curve)."""
    scores = []
    for pred in model.staged_predict(X_test):
        if is_classifier:
            scores.append(accuracy_score(y_test, pred))
        else:
            scores.append(r2_score(y_test, pred))
    return scores


def gb_classification():
    print("\n=== Gradient Boosting Classifier (Breast Cancer) ===")

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Effect of learning_rate × n_estimators interaction ---
    print("\n  learning_rate vs accuracy (n_estimators=200):")
    for lr in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
        gb = GradientBoostingClassifier(n_estimators=200, learning_rate=lr,
                                        max_depth=3, random_state=42)
        gb.fit(X_train, y_train)
        acc = accuracy_score(y_test, gb.predict(X_test))
        print(f"    lr={lr:.3f}: acc={acc:.4f}")

    # --- Staged prediction: overfitting detection ---
    gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05,
                                    max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    train_scores = staged_score(gb, X_train, y_train, True)
    test_scores  = staged_score(gb, X_test,  y_test,  True)

    best_n = np.argmax(test_scores) + 1
    print(f"\n  Best n_estimators (staged): {best_n}, test_acc={test_scores[best_n-1]:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(train_scores, label='Train')
    plt.plot(test_scores,  label='Test')
    plt.axvline(best_n - 1, color='r', linestyle='--', label=f'Best n={best_n}')
    plt.xlabel("Number of Trees"); plt.ylabel("Accuracy")
    plt.title("Gradient Boosting: Learning Curve"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

    # --- Compare to Random Forest ---
    print("\n  Gradient Boosting vs Random Forest (CV):")
    rf  = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_best = GradientBoostingClassifier(n_estimators=best_n, learning_rate=0.05,
                                          max_depth=3, random_state=42)
    for name, model in [('RandomForest', rf), ('GradientBoosting', gb_best)]:
        cv = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
        print(f"    {name:20s}: {cv.mean():.4f} ± {cv.std():.4f}")


def gb_regression():
    print("\n=== Gradient Boosting Regressor (California Housing) ===")

    X, y = fetch_california_housing(return_X_y=True)
    feature_names = fetch_california_housing().feature_names
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gb = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,       # stochastic GB: train each tree on 80% of data
        min_samples_leaf=10,
        random_state=42
    )
    gb.fit(X_train, y_train)

    print(f"\n  Train R²: {r2_score(y_train, gb.predict(X_train)):.4f}")
    print(f"  Test  R²: {r2_score(y_test,  gb.predict(X_test)):.4f}")

    print("\n  Feature importances:")
    for name, imp in sorted(zip(feature_names, gb.feature_importances_), key=lambda x: -x[1]):
        bar = '█' * int(imp * 40)
        print(f"    {name:25s}: {imp:.4f}  {bar}")

    # Compare RF vs GB for regression
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    print(f"\n  RF  test R²: {r2_score(y_test, rf.predict(X_test)):.4f}")
    print(f"  GB  test R²: {r2_score(y_test, gb.predict(X_test)):.4f}")
    print("  → GB usually wins, but takes longer to train and needs more tuning.")


def main():
    print("=" * 55)
    print("GRADIENT BOOSTING — Sequential Ensemble")
    print("=" * 55)
    gb_classification()
    gb_regression()


if __name__ == "__main__":
    main()
