"""t-SNE — map data while emphasizing local neighborhood relationships.

t-SNE converts pairwise similarities into probabilities, then optimizes a
low-dimensional embedding to match them. It helps explore high-dimensional data
in 2D/3D. Distances between far-apart islands, island sizes, and axis values do
not generally preserve global geometry; perplexity and random starts can change
the layout. Do not treat the plot as a clustering proof or its coordinates as
general-purpose features without a separate rationale.

IMPLEMENTATION: scale a subset for speed, fit_transform, and compare nearest-
neighbor overlap as one limited diagnostic of local structure preservation.
"""
import inspect
import numpy as np
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

X, labels_for_inspection = load_digits(return_X_y=True)
ix = np.random.default_rng(8).choice(len(X), size=500, replace=False)
X, labels_for_inspection = StandardScaler().fit_transform(X[ix]), labels_for_inspection[ix]
iteration_arg = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
embedding = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
                 random_state=8, **{iteration_arg: 750}).fit_transform(X)
def neighbor_sets(A, k=10):
    D = pairwise_distances(A)
    np.fill_diagonal(D, np.inf)
    return [set(np.argpartition(row, k)[:k]) for row in D]
before, after = neighbor_sets(X), neighbor_sets(embedding)
overlap = np.mean([len(a & b) / len(a) for a, b in zip(before, after)])
print("Embedding shape:", embedding.shape, "10-neighbor overlap:", round(float(overlap), 3))
print("First points (coordinates have no intrinsic units):", np.round(embedding[:5], 2))
print("Labels only help inspect the map; they were not used to fit it:", labels_for_inspection[:20].tolist())
