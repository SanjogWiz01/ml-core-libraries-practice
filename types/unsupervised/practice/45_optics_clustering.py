"""45. OPTICS: density clustering that does not need a fixed eps."""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score

X, _ = make_moons(n_samples=500, noise=0.06, random_state=42)

labels = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.1).fit_predict(X)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
mask = labels >= 0
print("clusters found:", n_clusters)
print("noise points:", int((labels == -1).sum()))
print("labelled counts:", np.bincount(labels[mask]))
print("silhouette (non-noise):", round(silhouette_score(X[mask], labels[mask]), 3))
print("OPTICS infers eps internally from reachability, unlike DBSCAN.")
