"""1. Basic K-Means clustering on unlabelled data."""
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)

model = KMeans(n_clusters=4, n_init=10, random_state=42)
labels = model.fit_predict(X)

print("first 20 labels:", labels[:20])
print("unique labels:", sorted(set(labels)))
print("points per cluster:", [int((labels == k).sum()) for k in range(4)])
