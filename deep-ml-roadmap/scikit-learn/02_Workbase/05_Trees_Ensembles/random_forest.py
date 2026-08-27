"""Random Forest — ensembles, feature importance, OOB score."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, r2_score


def random_forest_classification():
    print("\n=== Random Forest Classifier (Breast Cancer) ===")

    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Compare: single DT vs Random Forest ---
    print("\n  Single DT vs Random Forest:")
    dt = DecisionTreeClassifier(max_depth=None, random_state=42)
    dt.fit(X_train, y_train)
    print(f"    Decision Tree (deep):     acc={accuracy_score(y_test, dt.predict(X_test)):.4f}")

    rf = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
    rf.fit(X_train, y_train)
    print(f"    Random Forest (100 trees): acc={accuracy_score(y_test, rf.predict(X_test)):.4f}  "
          f"OOB={rf.oob_score_:.4f}")

    # --- Effect of n_estimators ---
    print("\n  n_estimators vs accuracy:")
    for n in [1, 5, 10, 25, 50, 100, 200]:
        r = RandomForestClassifier(n_estimators=n, random_state=42)
        r.fit(X_train, y_train)
        acc = accuracy_score(y_test, r.predict(X_test))
        print(f"    n={n:3d}: {acc:.4f}")

    # --- Feature importance ---
    rf_big = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_big.fit(X_train, y_train)
    importances = rf_big.feature_importances_

    top_n = 15
    indices = np.argsort(importances)[::-1][:top_n]

    print(f"\n  Top {top_n} feature importances:")
    for rank, i in enumerate(indices, 1):
        bar = '█' * int(importances[i] * 50)
        print(f"    {rank:2}. {data.feature_names[i]:35s}: {importances[i]:.4f}  {bar}")

    plt.figure(figsize=(10, 5))
    plt.bar(range(top_n), importances[indices], tick_label=[data.feature_names[i] for i in indices])
    plt.xticks(rotation=45, ha='right')
    plt.title("Random Forest Feature Importances")
    plt.ylabel("Mean Decrease in Impurity")
    plt.tight_layout()
    plt.show()

    # --- Cross-validation ---
    cv = cross_val_score(rf_big, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"\n  5-Fold CV accuracy: {cv.mean():.4f} ± {cv.std():.4f}")


def random_forest_regression():
    print("\n=== Random Forest Regressor (California Housing) ===")

    X, y = fetch_california_housing(return_X_y=True)
    feature_names = fetch_california_housing().feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        max_features=0.5,   # use 50% of features per split (prevents correlation between trees)
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    print(f"\n  Train R²: {r2_score(y_train, rf.predict(X_train)):.4f}")
    print(f"  Test  R²: {r2_score(y_test,  rf.predict(X_test)):.4f}")
    print(f"  OOB   R²: {rf.oob_score_:.4f}")

    print("\n  Feature importances:")
    for name, imp in sorted(zip(feature_names, rf.feature_importances_), key=lambda x: -x[1]):
        bar = '█' * int(imp * 40)
        print(f"    {name:25s}: {imp:.4f}  {bar}")


def main():
    print("=" * 55)
    print("RANDOM FOREST — Bagging Ensemble")
    print("=" * 55)
    random_forest_classification()
    random_forest_regression()


if __name__ == "__main__":
    main()
