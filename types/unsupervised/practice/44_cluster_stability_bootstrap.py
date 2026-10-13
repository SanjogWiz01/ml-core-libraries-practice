"""44. Cluster stability: resample the data and check agreement (ARI)."""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)
base = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(X)

rng = np.random.default_rng(42)
scores = []
for i in range(20):
    idx = rng.choice(len(X), size=len(X), replace=True)
    labels = KMeans(n_clusters=4, n_init=10, random_state=i).fit_predict(X[idx])
    scores.append(adjusted_rand_score(base[idx], labels))

scores = np.array(scores)
print("bootstrap ARI scores:", scores.round(3))
print(f"mean={scores.mean():.3f}  std={scores.std():.3f}")
print("high mean + low std => the clustering is stable under resampling.")
