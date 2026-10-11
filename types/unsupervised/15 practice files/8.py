"""8. DBSCAN: density-based clustering finds arbitrary shapes and noise."""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

X, _ = make_moons(n_samples=500, noise=0.06, random_state=42)

labels = DBSCAN(eps=0.2, min_samples=5).fit_predict(X)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = int((labels == -1).sum())
print("clusters found:", n_clusters)
print("noise points:", n_noise)
print("label counts:", np.bincount(labels[labels >= 0]))
