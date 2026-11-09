"""ELBOW METHOD — inspect diminishing returns as the K-Means k grows.

Inertia cannot increase as k rises, so minimum inertia alone picks too many
clusters. An elbow is a bend where extra groups reduce inertia less sharply.
Some data has no distinct bend; the result is a heuristic, not a true-k test.
Combine it with silhouette, stability, domain needs, and practical usefulness.

IMPLEMENTATION: each loop fits the same observations with a different k and
collects inertia. The successive reductions make a possible bend visible.
"""
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=450, centers=4, cluster_std=.85, random_state=21)
inertias = []
for k in range(1, 9):
    fitted = KMeans(n_clusters=k, n_init=15, random_state=21).fit(X)
    inertias.append(float(fitted.inertia_))
print("k       :", list(range(1, 9)))
print("inertia :", [round(v, 1) for v in inertias])
print("reduction:", [None] + [round(inertias[i-1]-inertias[i], 1) for i in range(1, len(inertias))])
print("Look for where reductions flatten; there is no guaranteed elbow.")
