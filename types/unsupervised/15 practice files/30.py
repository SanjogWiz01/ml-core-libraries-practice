"""30. Affinity Propagation: message passing, no K needed, finds exemplars."""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

model = AffinityPropagation(damping=0.9, preference=-50, random_state=42).fit(X)
labels = model.labels_
n_clusters = len(np.unique(labels))

print("clusters found:", n_clusters)
print("cluster counts:", np.bincount(labels))
print("exemplar indices (actual data points):", model.cluster_centers_indices_[:10])
print("silhouette:", round(silhouette_score(X, labels), 3))
