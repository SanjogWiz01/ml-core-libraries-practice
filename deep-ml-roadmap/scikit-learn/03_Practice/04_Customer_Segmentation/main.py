"""Customer Segmentation — K-Means, DBSCAN, PCA visualization."""

from pathlib import Path
import subprocess, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).parent.parent.parent
DATA_PATH = ROOT / "data" / "raw" / "customer_segmentation.csv"


def ensure_data():
    if not DATA_PATH.exists():
        subprocess.run([sys.executable, str(ROOT / "data" / "raw" / "generate_datasets.py")],
                       check=True)


def load_and_explore(path):
    df = pd.read_csv(path)
    print(f"\nDataset: {df.shape}")
    print(f"Missing: {df.isnull().sum().sum()}")
    print(f"\nNumeric stats:\n{df.describe().round(2)}")
    return df


def preprocess(df):
    numeric_cols  = ['annual_income', 'spending_score', 'age', 'num_purchases', 'avg_order_value']
    nominal_cols  = ['gender', 'region']

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler())
        ]), numeric_cols),
        ('nom', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe',     OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ]), nominal_cols),
    ])

    X = df[numeric_cols + nominal_cols]
    X_scaled = preprocessor.fit_transform(X)
    return X_scaled, numeric_cols, preprocessor


def find_optimal_k(X_scaled):
    print("\n--- Finding optimal K (elbow + silhouette) ---")
    inertias, silhouettes = [], []
    K_range = range(2, 11)

    for k in K_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        print(f"  K={k}: inertia={km.inertia_:.1f}, silhouette={silhouettes[-1]:.4f}")

    best_k = list(K_range)[np.argmax(silhouettes)]
    print(f"\nBest K: {best_k}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(K_range, inertias, 'bo-')
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method"); axes[0].grid(alpha=0.3)

    axes[1].plot(K_range, silhouettes, 'rs-')
    axes[1].axvline(best_k, color='k', linestyle='--', label=f'Best K={best_k}')
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.show()

    return best_k


def kmeans_segmentation(df, X_scaled, best_k):
    print(f"\n--- K-Means (K={best_k}) ---")
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)

    df = df.copy()
    df['cluster'] = labels

    print(f"\nCluster sizes:\n{df['cluster'].value_counts().sort_index()}")
    print(f"\nCluster profiles (numeric means):")
    numeric_cols = ['annual_income', 'spending_score', 'age', 'num_purchases', 'avg_order_value']
    print(df.groupby('cluster')[numeric_cols].mean().round(1).to_string())

    sil = silhouette_score(X_scaled, labels)
    print(f"\nSilhouette score: {sil:.4f}")
    return df, labels


def dbscan_segmentation(X_scaled):
    print("\n--- DBSCAN ---")
    for eps in [0.3, 0.5, 0.8, 1.0]:
        db = DBSCAN(eps=eps, min_samples=10)
        labels = db.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise    = (labels == -1).sum()
        if n_clusters >= 2:
            sil = silhouette_score(X_scaled[labels != -1], labels[labels != -1])
            print(f"  eps={eps}: clusters={n_clusters}, noise={n_noise}, sil={sil:.4f}")
        else:
            print(f"  eps={eps}: clusters={n_clusters}, noise={n_noise}")


def visualize_clusters(X_scaled, labels, title):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    var = pca.explained_variance_ratio_
    unique_labels = sorted(set(labels))
    colors = plt.cm.Set1(np.linspace(0, 0.9, max(len(unique_labels), 2)))

    plt.figure(figsize=(8, 6))
    for k, col in zip(unique_labels, colors):
        mask = labels == k
        label = f'Cluster {k}' if k != -1 else 'Noise'
        c = 'gray' if k == -1 else col
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=15, alpha=0.6, c=[c], label=label)
    plt.xlabel(f"PC1 ({var[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({var[1]*100:.1f}%)")
    plt.title(title)
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


def main():
    print("=" * 55)
    print("CUSTOMER SEGMENTATION — Unsupervised Learning")
    print("=" * 55)

    ensure_data()
    df = load_and_explore(DATA_PATH)
    X_scaled, numeric_cols, preprocessor = preprocess(df)

    print(f"\nPreprocessed shape: {X_scaled.shape}")

    best_k = find_optimal_k(X_scaled)
    df_clustered, km_labels = kmeans_segmentation(df, X_scaled, best_k)
    dbscan_segmentation(X_scaled)

    visualize_clusters(X_scaled, km_labels, f"Customer Segments — K-Means (K={best_k})")

    # DBSCAN visualization
    db = DBSCAN(eps=0.5, min_samples=10)
    db_labels = db.fit_predict(X_scaled)
    visualize_clusters(X_scaled, db_labels, "Customer Segments — DBSCAN")

    print("\n--- Segment Interpretation ---")
    print("  Use cluster profiles to name segments, e.g.:")
    print("  High income + high spending → Premium customers")
    print("  Low income + high spending  → Young spenders")
    print("  High income + low spending  → Conservative big earners")
    print("  Low income + low spending   → Budget customers")


if __name__ == "__main__":
    main()
