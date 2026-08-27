"""K-Means and DBSCAN clustering with evaluation and visualization."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def kmeans_demo():
    print("\n=== K-Means Clustering ===")

    X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # --- Elbow method ---
    print("\n  Elbow method (inertia vs K):")
    inertias, sil_scores = [], []
    K_range = range(1, 11)

    for k in K_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X_s)
        inertias.append(km.inertia_)
        if k >= 2:
            sil = silhouette_score(X_s, km.labels_)
            sil_scores.append(sil)
            print(f"    K={k}: inertia={km.inertia_:.1f}, silhouette={sil:.4f}")
        else:
            print(f"    K={k}: inertia={km.inertia_:.1f}")

    # Best K by silhouette
    best_k = list(range(2, 11))[np.argmax(sil_scores)]
    print(f"\n  Best K by silhouette: {best_k}")

    # Elbow + silhouette plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(K_range, inertias, 'bo-')
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia (WCSS)")
    axes[0].set_title("Elbow Method"); axes[0].grid(alpha=0.3)

    axes[1].plot(range(2, 11), sil_scores, 'rs-')
    axes[1].axvline(best_k, color='k', linestyle='--', label=f'Best K={best_k}')
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.show()

    # --- Fit with best K ---
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    km.fit(X_s)
    labels = km.labels_
    centers = km.cluster_centers_

    sil = silhouette_score(X_s, labels)
    print(f"\n  Fitted K-Means (K={best_k}):")
    print(f"    Inertia:         {km.inertia_:.2f}")
    print(f"    Silhouette:      {sil:.4f}")
    print(f"    Cluster sizes:   {np.bincount(labels)}")

    # Scatter plot
    plt.figure(figsize=(7, 5))
    colors = ['steelblue', 'firebrick', 'forestgreen', 'darkorange', 'purple']
    for k in range(best_k):
        mask = labels == k
        plt.scatter(X_s[mask, 0], X_s[mask, 1], s=20, alpha=0.6,
                    color=colors[k % len(colors)], label=f'Cluster {k}')
    plt.scatter(centers[:, 0], centers[:, 1], s=200, c='black', marker='X',
                zorder=5, label='Centroids')
    plt.title(f"K-Means (K={best_k})")
    plt.legend(); plt.tight_layout(); plt.show()


def dbscan_demo():
    print("\n=== DBSCAN Clustering ===")

    # Moon-shaped data — K-Means fails here
    X_moons, _ = make_moons(n_samples=300, noise=0.08, random_state=42)

    # Blobs with outliers
    X_blobs, _ = make_blobs(n_samples=200, centers=3, random_state=42)
    n_outliers = 30
    rng = np.random.default_rng(42)
    outliers = rng.uniform(-6, 6, (n_outliers, 2))
    X_out = np.vstack([X_blobs, outliers])

    scaler = StandardScaler()

    datasets = [
        ("Moon-shaped data",         X_moons,  0.2,  5),
        ("Blobs + outliers",         X_out,    0.5, 5),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, (title, X, eps, min_samples) in enumerate(datasets):
        X_s = scaler.fit_transform(X)

        # K-Means
        km = KMeans(n_clusters=2 if 'Moon' in title else 3, n_init=10, random_state=42)
        km.fit(X_s)

        # DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples)
        db.fit(X_s)
        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        n_noise    = (db.labels_ == -1).sum()

        print(f"\n  {title}:")
        print(f"    DBSCAN (eps={eps}, min_samples={min_samples}): "
              f"{n_clusters} clusters, {n_noise} noise points")
        if n_clusters >= 2:
            mask_valid = db.labels_ != -1
            sil = silhouette_score(X_s[mask_valid], db.labels_[mask_valid])
            print(f"    Silhouette (excluding noise): {sil:.4f}")

        # Plot K-Means
        axes[row, 0].scatter(X_s[:, 0], X_s[:, 1], c=km.labels_, cmap='Set1', s=15, alpha=0.7)
        axes[row, 0].set_title(f"K-Means — {title}")

        # Plot DBSCAN
        colors = plt.cm.Set1(np.linspace(0, 1, max(1, n_clusters)))
        for k in set(db.labels_):
            mask = db.labels_ == k
            color = 'gray' if k == -1 else colors[k % len(colors)]
            label = 'Noise' if k == -1 else f'Cluster {k}'
            axes[row, 1].scatter(X_s[mask, 0], X_s[mask, 1], c=[color], s=15, alpha=0.7, label=label)
        axes[row, 1].set_title(f"DBSCAN — {title}")
        axes[row, 1].legend(fontsize=7, loc='upper right')

    plt.tight_layout(); plt.show()

    # --- eps sensitivity ---
    print("\n  DBSCAN eps sensitivity (moon data):")
    X_s = StandardScaler().fit_transform(X_moons)
    for eps in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        db = DBSCAN(eps=eps, min_samples=5)
        db.fit(X_s)
        n_c = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        n_n = (db.labels_ == -1).sum()
        print(f"    eps={eps:.2f}: clusters={n_c}, noise={n_n}")


def main():
    print("=" * 55)
    print("CLUSTERING — K-Means and DBSCAN")
    print("=" * 55)
    kmeans_demo()
    dbscan_demo()

    print("\n--- Choosing Between K-Means and DBSCAN ---")
    print("  K-Means:  convex/spherical clusters, K is known, large data, fast")
    print("  DBSCAN:   arbitrary shapes, K unknown, outlier detection needed, smaller data")


if __name__ == "__main__":
    main()
