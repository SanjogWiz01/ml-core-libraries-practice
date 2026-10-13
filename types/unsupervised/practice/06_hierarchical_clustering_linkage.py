"""6. Hierarchical (agglomerative) clustering - no K guess via linkage tree."""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)

for linkage in ("ward", "complete", "average", "single"):
    labels = AgglomerativeClustering(n_clusters=4, linkage=linkage).fit_predict(X)
    print(linkage, "counts:", np.bincount(labels),
          "silhouette:", round(silhouette_score(X, labels), 3))
