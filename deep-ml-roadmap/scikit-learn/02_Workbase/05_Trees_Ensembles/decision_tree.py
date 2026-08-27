"""Decision Trees — classification and regression, depth control, visualization."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, r2_score


def decision_tree_classifier():
    print("\n=== Decision Tree Classifier (Iris) ===")

    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Effect of max_depth on overfitting ---
    print("\n  max_depth vs accuracy (train / test):")
    depths = [1, 2, 3, 4, 5, 10, None]
    train_accs, test_accs = [], []

    for depth in depths:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        tr = accuracy_score(y_train, dt.predict(X_train))
        te = accuracy_score(y_test,  dt.predict(X_test))
        train_accs.append(tr)
        test_accs.append(te)
        depth_str = str(depth) if depth else 'None'
        print(f"    depth={depth_str:4}: train={tr:.4f}, test={te:.4f}", end='')
        if tr - te > 0.05:
            print("  ← overfitting")
        else:
            print()

    # --- Best depth model ---
    best_depth = depths[np.argmax(test_accs)]
    dt = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    dt.fit(X_train, y_train)

    print(f"\n  Best depth: {best_depth}")
    print(f"  Test accuracy: {accuracy_score(y_test, dt.predict(X_test)):.4f}")

    # --- Feature importance ---
    print("\n  Feature importance:")
    for name, imp in sorted(
        zip(data.feature_names, dt.feature_importances_), key=lambda x: -x[1]
    ):
        bar = '█' * int(imp * 30)
        print(f"    {name:25s}: {imp:.4f}  {bar}")

    # --- Visualize tree ---
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(dt, feature_names=data.feature_names, class_names=data.target_names,
              filled=True, rounded=True, ax=ax)
    ax.set_title(f"Decision Tree (max_depth={best_depth})")
    plt.tight_layout()
    plt.show()

    # --- depth vs accuracy plot ---
    depth_labels = [str(d) if d else 'None' for d in depths]
    x = range(len(depths))
    plt.figure(figsize=(8, 4))
    plt.plot(x, train_accs, 'bo-', label='Train')
    plt.plot(x, test_accs,  'rs-', label='Test')
    plt.xticks(x, depth_labels)
    plt.xlabel("max_depth"); plt.ylabel("Accuracy")
    plt.title("Decision Tree: Depth vs Accuracy")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

    return dt


def decision_tree_regressor():
    print("\n=== Decision Tree Regressor (California Housing) ===")

    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Compare depth and min_samples_leaf
    print("\n  Depth and min_samples_leaf effect:")
    for depth in [3, 5, 10, None]:
        for min_leaf in [1, 10, 50]:
            dt = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_leaf, random_state=42)
            dt.fit(X_train, y_train)
            r2 = r2_score(y_test, dt.predict(X_test))
            d_str = str(depth) if depth else 'None'
            print(f"    depth={d_str:4}, min_leaf={min_leaf:2}: R²={r2:.4f}", end='')
            train_r2 = r2_score(y_train, dt.predict(X_train))
            if train_r2 - r2 > 0.15:
                print("  ← overfitting")
            else:
                print()

    # Best configuration
    best = DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=42)
    best.fit(X_train, y_train)
    print(f"\n  Best config (depth=8, min_leaf=20): R²={r2_score(y_test, best.predict(X_test)):.4f}")


def main():
    print("=" * 55)
    print("DECISION TREES — Classifier & Regressor")
    print("=" * 55)
    decision_tree_classifier()
    decision_tree_regressor()


if __name__ == "__main__":
    main()
