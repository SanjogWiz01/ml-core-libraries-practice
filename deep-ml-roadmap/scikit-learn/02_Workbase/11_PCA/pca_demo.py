"""PCA — dimensionality reduction, explained variance, visualization."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")


def pca_explained_variance():
    print("\n=== PCA Explained Variance ===")

    data = load_breast_cancer()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Fit PCA with all components
    pca = PCA()
    pca.fit(X_s)

    explained = pca.explained_variance_ratio_
    cumulative = explained.cumsum()

    print(f"\n  Dataset: {X.shape[0]} samples × {X.shape[1]} features")
    print("\n  Explained variance per component:")
    for i, (ev, cum) in enumerate(zip(explained[:10], cumulative[:10])):
        bar = '█' * int(ev * 100)
        print(f"    PC{i+1:2d}: {ev:.4f}  cumulative={cum:.4f}  {bar}")

    # How many components for 90%, 95%, 99%?
    for threshold in [0.90, 0.95, 0.99]:
        n = (cumulative < threshold).sum() + 1
        print(f"\n  → {n} components explain {threshold*100:.0f}% of variance")

    # Scree plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(1, len(explained)+1), explained)
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].set_title("Scree Plot")
    axes[0].set_xlim(0.5, 20.5)  # show first 20

    axes[1].plot(range(1, len(cumulative)+1), cumulative, 'bo-')
    for t in [0.90, 0.95, 0.99]:
        n = (cumulative < t).sum() + 1
        axes[1].axhline(t, color='r', linestyle='--', linewidth=0.8)
        axes[1].axvline(n, color='r', linestyle=':', linewidth=0.8)
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance")
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.show()


def pca_2d_visualization():
    print("\n=== PCA 2D Visualization ===")

    data = load_breast_cancer()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    pca2 = PCA(n_components=2)
    X_2d = pca2.fit_transform(X_s)

    print(f"\n  Original: {X.shape[1]} dims → PCA: 2 dims")
    print(f"  Variance retained: {pca2.explained_variance_ratio_.sum()*100:.1f}%")

    plt.figure(figsize=(8, 6))
    for label, name, color in zip([0, 1], data.target_names, ['firebrick', 'steelblue']):
        mask = y == label
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=20, alpha=0.6,
                    label=name, color=color)
    plt.xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("PCA — Breast Cancer (2D projection)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


def pca_as_preprocessing():
    print("\n=== PCA as Preprocessing — Performance vs n_components ===")

    # Digits dataset: 64 features
    data = load_digits()
    X, y = data.data, data.target
    print(f"\n  Digits dataset: {X.shape[0]} samples × {X.shape[1]} features, {len(np.unique(y))} classes")

    results = {}
    for n in [2, 5, 10, 20, 30, 40, 64]:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca',   PCA(n_components=n)),
            ('model', LogisticRegression(max_iter=1000, random_state=42))
        ])
        cv = cross_val_score(pipe, X, y, cv=5, scoring='accuracy', n_jobs=-1)
        results[n] = cv.mean()
        print(f"  n_components={n:2d}: CV accuracy={cv.mean():.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(list(results.keys()), list(results.values()), 'bo-')
    plt.xlabel("n_components"); plt.ylabel("CV Accuracy")
    plt.title("PCA n_components vs Model Accuracy (Digits)")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


def pca_components_interpretation():
    print("\n=== PCA Components — What Are They? ===")

    data = load_breast_cancer()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    pca = PCA(n_components=3)
    pca.fit(X_s)

    feature_names = data.feature_names
    print(f"\n  PC1 (explains {pca.explained_variance_ratio_[0]*100:.1f}%):")
    top5 = np.argsort(np.abs(pca.components_[0]))[::-1][:5]
    for i in top5:
        print(f"    {feature_names[i]:35s}: {pca.components_[0][i]:+.4f}")

    print(f"\n  PC2 (explains {pca.explained_variance_ratio_[1]*100:.1f}%):")
    top5 = np.argsort(np.abs(pca.components_[1]))[::-1][:5]
    for i in top5:
        print(f"    {feature_names[i]:35s}: {pca.components_[1][i]:+.4f}")

    print("\n  → Components are linear combinations of original features.")
    print("  → High loadings (positive or negative) indicate important features.")
    print("  → Components are orthogonal — they capture independent variance.")


def main():
    print("=" * 55)
    print("PCA — Principal Component Analysis")
    print("=" * 55)
    pca_explained_variance()
    pca_2d_visualization()
    pca_as_preprocessing()
    pca_components_interpretation()


if __name__ == "__main__":
    main()
