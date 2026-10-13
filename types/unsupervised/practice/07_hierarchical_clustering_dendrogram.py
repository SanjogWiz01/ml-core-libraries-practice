"""7. Dendrogram: visualise the merge order of hierarchical clustering."""
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=50, centers=4, cluster_std=0.8, random_state=42)

Z = linkage(X, method="ward")

plt.figure(figsize=(10, 5))
dendrogram(Z, no_labels=True)
plt.title("Ward linkage dendrogram")
plt.xlabel("samples")
plt.ylabel("merge distance")
plt.show()

print("last 3 merges (left, right, distance, count):")
print(Z[-3:].round(2))
