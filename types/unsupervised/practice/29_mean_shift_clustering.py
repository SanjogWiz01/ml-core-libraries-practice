"""29. Mean Shift: discover clusters without fixing K in advance."""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=800, centers=5, cluster_std=0.8, random_state=42)

bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
model = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X)

n_clusters = len(np.unique(model.labels_))
print("estimated bandwidth:", round(bandwidth, 3))
print("clusters found:", n_clusters)
print("cluster counts:", np.bincount(model.labels_))
print("silhouette:", round(silhouette_score(X, model.labels_), 3))
