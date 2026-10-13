"""27. Structured clustering: agglomerative with connectivity constraints."""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score

X, _ = make_moons(n_samples=500, noise=0.06, random_state=42)

plain = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(X)

graph = kneighbors_graph(X, n_neighbors=10, include_self=False)
connected = AgglomerativeClustering(
    n_clusters=2, linkage="ward", connectivity=graph
).fit_predict(X)

print("plain ward          counts:", np.bincount(plain),
      "silhouette:", round(silhouette_score(X, plain), 3))
print("connectivity ward   counts:", np.bincount(connected),
      "silhouette:", round(silhouette_score(X, connected), 3))
