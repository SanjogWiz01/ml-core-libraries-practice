"""38. Profiling clusters: interpret centroids back in original units."""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
income = np.r_[rng.normal(30_000, 5_000, 200),
               rng.normal(70_000, 8_000, 200),
               rng.normal(110_000, 10_000, 200)]
age = np.r_[rng.normal(25, 3, 200),
            rng.normal(40, 5, 200),
            rng.normal(55, 4, 200)]
X = np.column_stack([income, age])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X_scaled)
centroids_scaled = km.cluster_centers_
centroids = scaler.inverse_transform(centroids_scaled)

order = np.argsort(-centroids[:, 0])
print("cluster profiles (sorted by income, original units):")
for c in order:
    size = int((km.labels_ == c).sum())
    print(f"  cluster {c}: n={size:<4} avg income={centroids[c, 0]:>10,.0f}  "
          f"avg age={centroids[c, 1]:>5.1f}")
