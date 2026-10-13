"""3. Elbow method: pick K by looking for the bend in inertia."""
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)

ks = range(1, 11)
inertias = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_ for k in ks]

print(list(zip(ks, [round(v, 1) for v in inertias])))

plt.plot(list(ks), inertias, marker="o")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow method")
plt.show()
