"""26. K-Means initialisation: k-means++ vs random start."""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=500, centers=8, cluster_std=1.5, random_state=42)

for init in ("k-means++", "random"):
    model = KMeans(n_clusters=8, init=init, n_init=1, random_state=0).fit(X)
    print(f"init={init:<10} inertia={model.inertia_:9.1f}  iterations={model.n_iter_}")

random_runs = [
    KMeans(n_clusters=8, init="random", n_init=1, random_state=s).fit(X).inertia_
    for s in range(10)
]
print("random init inertia spread: min={:.0f} max={:.0f} std={:.0f}".format(
    min(random_runs), max(random_runs), np.std(random_runs)))
print("k-means++ is deterministic-ish and usually avoids bad local minima.")
