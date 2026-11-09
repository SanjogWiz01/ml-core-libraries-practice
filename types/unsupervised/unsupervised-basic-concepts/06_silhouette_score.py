"""SILHOUETTE SCORE — compare within-cluster cohesion to the next group.

For point i, a(i) is its mean distance to its own cluster. b(i) is the lowest
mean distance to another cluster. Its score is (b-a)/max(a,b), in [-1, 1].
Positive suggests separation, near zero overlap, negative possible misassignment.
The mean hides weak clusters, depends on distance, and is defined only with
2..n-1 labels. Silhouette favors compact separated groups.

IMPLEMENTATION: evaluate candidate K-Means partitions using a common metric;
sample-level values let us inspect each cluster as well as the overall average.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

X, _ = make_blobs(n_samples=350, centers=4, cluster_std=1.1, random_state=5)
X = StandardScaler().fit_transform(X)
for k in (2, 3, 4, 5, 6):
    labels = KMeans(n_clusters=k, n_init=15, random_state=5).fit_predict(X)
    samples = silhouette_samples(X, labels)
    per_cluster = [round(float(samples[labels == c].mean()), 3) for c in range(k)]
    print(f"k={k}: overall={silhouette_score(X, labels):.3f}; per-cluster={per_cluster}")
