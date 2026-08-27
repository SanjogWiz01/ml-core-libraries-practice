"""StandardScaler, MinMaxScaler, RobustScaler — when and why to use each."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def compare_scalers(X_train, X_test, y_train, y_test):
    """Train KNN with each scaler and compare accuracy (KNN is scale-sensitive)."""
    scalers = {
        'No scaling':     None,
        'StandardScaler': StandardScaler(),
        'MinMaxScaler':   MinMaxScaler(),
        'RobustScaler':   RobustScaler(),
    }

    print("\n--- KNN accuracy with different scalers ---")
    for name, scaler in scalers.items():
        if scaler is None:
            Xtr, Xte = X_train, X_test
        else:
            Xtr = scaler.fit_transform(X_train)
            Xte = scaler.transform(X_test)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(Xtr, y_train)
        acc = knn.score(Xte, y_test)
        print(f"  {name:20s}: {acc:.4f}")


def visualize_scaling(X, feature_idx=0):
    """Show distribution of one feature before and after scaling."""
    x = X[:, feature_idx:feature_idx+1]

    scalers = {
        'Original':      x,
        'Standard':      StandardScaler().fit_transform(x),
        'MinMax [0,1]':  MinMaxScaler().fit_transform(x),
        'Robust':        RobustScaler().fit_transform(x),
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (name, data) in zip(axes, scalers.items()):
        ax.hist(data.ravel(), bins=30, edgecolor='white')
        ax.set_title(name)
        ax.set_ylabel("Count")
    fig.suptitle(f"Feature scaling comparison (feature index {feature_idx})")
    plt.tight_layout()
    plt.show()


def demonstrate_outlier_sensitivity(n=200, seed=42):
    """Show how outliers affect MinMax vs RobustScaler."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)

    # Add outliers
    x_with_outliers = np.append(x, [15, -12, 20])

    ss   = StandardScaler()
    mms  = MinMaxScaler()
    rs   = RobustScaler()

    print("\n--- Outlier effect on scalers ---")
    print("Data range without outliers: [{:.1f}, {:.1f}]".format(x.min(), x.max()))
    print("Data range with outliers:    [{:.1f}, {:.1f}]".format(
        x_with_outliers.min(), x_with_outliers.max()))

    x2d = x_with_outliers.reshape(-1, 1)
    for name, scaler in [('StandardScaler', ss), ('MinMaxScaler', mms), ('RobustScaler', rs)]:
        scaled = scaler.fit_transform(x2d).ravel()
        # What is the "typical" value 0 mapped to?
        point_at_zero = scaler.transform([[0]]).ravel()[0]
        print(f"  {name}: scaled range=[{scaled.min():.2f}, {scaled.max():.2f}], "
              f"x=0 → {point_at_zero:.2f}")


def main():
    print("=" * 55)
    print("FEATURE SCALING — StandardScaler / MinMax / Robust")
    print("=" * 55)

    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    compare_scalers(X_train, X_test, y_train, y_test)
    visualize_scaling(X, feature_idx=0)
    demonstrate_outlier_sensitivity()

    # --- StandardScaler: what it stores ---
    print("\n--- StandardScaler internals ---")
    ss = StandardScaler()
    ss.fit(X_train)
    print(f"  mean_ shape: {ss.mean_.shape}")
    print(f"  First 3 feature means:  {ss.mean_[:3].round(4)}")
    print(f"  First 3 feature stds:   {ss.scale_[:3].round(4)}")
    print("  (These are computed from X_train only — test gets transformed with train stats)")

    # --- Important: fit on train, transform test ---
    X_train_s = ss.fit_transform(X_train)
    X_test_s  = ss.transform(X_test)   # uses X_train mean/std

    print(f"\n  Scaled train: mean≈{X_train_s.mean():.4f}, std≈{X_train_s.std():.4f}")
    print(f"  Scaled test:  mean≈{X_test_s.mean():.4f} (not exactly 0 — that's correct)")


if __name__ == "__main__":
    main()
