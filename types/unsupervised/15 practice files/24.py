"""24. Spectral clustering: graph-based grouping of non-convex shapes."""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score

X, _ = make_moons(n_samples=500, noise=0.06, random_state=42)

for name, model in [
    ("KMeans", KMeans(n_clusters=2, n_init=10, random_state=42)),
    ("Spectral (rbf)", SpectralClustering(n_clusters=2, affinity="rbf",
                                          gamma=10, random_state=42)),
    ("Spectral (knn)", SpectralClustering(n_clusters=2, affinity="nearest_neighbors",
                                          n_neighbors=10, random_state=42)),
]:
    labels = model.fit_predict(X)
    print(f"{name:<16} counts={np.bincount(labels)}  "
          f"silhouette={silhouette_score(X, labels):.3f}")

print("K-Means splits the moons in half; spectral follows the shape.")
