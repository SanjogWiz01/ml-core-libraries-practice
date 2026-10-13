"""51. Silhouette diagram: visualise per-cluster separation."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)

labels = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(X)
average = silhouette_score(X, labels)
samples = silhouette_samples(X, labels)

print("average silhouette:", round(average, 3))
for k in range(4):
    print(f"cluster {k}: mean silhouette={samples[labels == k].mean():.3f}")

plt.figure(figsize=(8, 5))
y_lower = 10
for k in range(4):
    values = np.sort(samples[labels == k])
    y_upper = y_lower + len(values)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, values, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * len(values), f"cluster {k}")
    y_lower = y_upper + 10

plt.axvline(x=average, color="red", linestyle="--", label="average")
plt.xlabel("silhouette coefficient")
plt.ylabel("samples grouped by cluster")
plt.title("Silhouette diagram")
plt.legend()
plt.show()
