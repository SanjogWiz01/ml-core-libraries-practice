"""UMAP — embed a neighborhood graph to reveal manifold structure.

UMAP estimates local neighborhoods, builds a fuzzy graph, and optimizes a low-
dimensional layout with a related graph structure. n_neighbors controls local
versus broader structure; min_dist influences how tightly points can pack. UMAP
can transform new points after fitting. Like t-SNE, it distorts some global
distances, and visible islands do not prove natural clusters. It is stochastic
and parameter-sensitive; set random_state for reproducibility (possibly slower).

IMPLEMENTATION: scale digits and fit UMAP with explicit settings; compare input
and output neighbor sets. Install optional dependency with `pip install umap-learn`.
"""
try:
    import umap
except ImportError:
    raise SystemExit("UMAP is optional. Install it with: python -m pip install umap-learn")
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

X, labels_for_inspection = load_digits(return_X_y=True)
X = StandardScaler().fit_transform(X[:500])
labels_for_inspection = labels_for_inspection[:500]
mapper = umap.UMAP(n_components=2, n_neighbors=15, min_dist=.1,
                   random_state=23, n_jobs=1)
Y = mapper.fit_transform(X)
def neighbors(A, k=10):
    D = pairwise_distances(A)
    np.fill_diagonal(D, np.inf)
    return [set(np.argpartition(row, k)[:k]) for row in D]
A, B = neighbors(X), neighbors(Y)
overlap = np.mean([len(a & b) / len(a) for a, b in zip(A, B)])
print("Embedding shape:", Y.shape, "10-neighbor overlap:", round(float(overlap), 3))
print("Labels only for post-fit inspection:", labels_for_inspection[:20].tolist())
