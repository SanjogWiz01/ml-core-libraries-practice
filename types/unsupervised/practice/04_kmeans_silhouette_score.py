"""4. Silhouette score: how well-separated the clusters are."""
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)

scores = {}
for k in range(2, 9):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    scores[k] = round(silhouette_score(X, labels), 3)

print(scores)
print("best K:", max(scores, key=scores.get))
