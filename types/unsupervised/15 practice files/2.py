"""2. Centroids, inertia and predicting new points."""
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)

model = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)

print("cluster centers:")
print(model.cluster_centers_.round(2))
print("inertia (within-cluster sum of squares):", round(model.inertia_, 2))

new_points = [[0, 0], [5, 5], [-6, -6]]
print("predictions for new points:", model.predict(new_points))
