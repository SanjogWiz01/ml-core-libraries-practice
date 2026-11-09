"""DISTANCE METRICS — choose what 'similar' means before clustering.

Euclidean measures straight-line displacement; Manhattan sums absolute changes;
cosine compares direction and ignores magnitude, often useful for normalized
text vectors; Hamming counts differing categorical positions. Feature scaling,
encoding, missingness, and the metric are one joint choice. In high dimensions,
many distances become less discriminative.

IMPLEMENTATION: pairwise_distances compares Euclidean, Manhattan, and cosine on
the same points. The first column has a much larger numeric scale; standardizing
changes the Euclidean geometry. Proportional text count vectors have zero cosine
distance despite differing magnitude.
"""
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler, normalize

X = np.array([[1., 1000.], [2., 1001.], [8., 1002.]])
for metric in ("euclidean", "manhattan", "cosine"):
    print(f"{metric:10s} distances from row 0:",
          np.round(pairwise_distances(X[:1], X[1:], metric=metric)[0], 4))
Z = StandardScaler().fit_transform(X)
print("Euclidean after standard scaling:", np.round(pairwise_distances(Z[:1], Z[1:])[0], 4))
texts = np.array([[3, 0, 1, 0], [30, 0, 10, 0], [0, 2, 0, 4]], dtype=float)
print("Cosine on normalized counts:\n", np.round(pairwise_distances(normalize(texts), metric="cosine"), 3))
