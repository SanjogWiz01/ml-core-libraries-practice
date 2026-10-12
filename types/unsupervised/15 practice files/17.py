"""17. MiniBatchKMeans: clustering that scales to large datasets."""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=100_000, centers=6, cluster_std=1.0, random_state=42)

full = KMeans(n_clusters=6, n_init=10, random_state=42).fit(X)
mini = MiniBatchKMeans(n_clusters=6, n_init=10, batch_size=1024, random_state=42).fit(X)

print("KMeans inertia:      ", round(full.inertia_, 1))
print("MiniBatch inertia:   ", round(mini.inertia_, 1))
print("KMeans silhouette:   ", round(silhouette_score(X, full.labels_, sample_size=5000, random_state=42), 3))
print("MiniBatch silhouette:", round(silhouette_score(X, mini.labels_, sample_size=5000, random_state=42), 3))
print("MiniBatch is much faster on large data with near-equal quality.")
