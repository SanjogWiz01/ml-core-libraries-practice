"""28. BIRCH: incremental clustering with a CF-tree, supports partial_fit."""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=2000, centers=5, cluster_std=0.7, random_state=42)

birch = Birch(n_clusters=5).fit(X)
labels = birch.predict(X)
print("batch fit  counts:", np.bincount(labels),
      "silhouette:", round(silhouette_score(X, labels), 3))

stream = Birch(n_clusters=5)
for batch in np.array_split(X, 10):          # online / streaming updates
    stream.partial_fit(batch)
labels_stream = stream.predict(X)
print("stream fit counts:", np.bincount(labels_stream),
      "silhouette:", round(silhouette_score(X, labels_stream), 3))
