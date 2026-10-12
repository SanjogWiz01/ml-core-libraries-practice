"""16. Cluster validity without labels: silhouette, Davies-Bouldin, Calinski-Harabasz."""
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)

labels = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(X)

print("silhouette (higher better):       ", round(silhouette_score(X, labels), 3))
print("davies-bouldin (lower better):    ", round(davies_bouldin_score(X, labels), 3))
print("calinski-harabasz (higher better):", round(calinski_harabasz_score(X, labels), 1))

# A deliberately bad K for comparison
bad = KMeans(n_clusters=9, n_init=10, random_state=42).fit_predict(X)
print("K=9 silhouette:", round(silhouette_score(X, bad), 3))
