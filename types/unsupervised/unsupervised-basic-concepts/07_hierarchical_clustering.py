"""HIERARCHICAL CLUSTERING — build nested groups from pairwise relationships.

Agglomerative clustering starts with one group per point and repeatedly merges
the closest groups. Single linkage uses the closest cross-pair and can chain;
complete uses the farthest pair; average uses mean cross-pair distance; Ward
minimizes the increase in within-group variance (Euclidean only). The merge tree
can be cut at a chosen level. Pairwise memory/time often limits large datasets.

IMPLEMENTATION: scipy linkage encodes each merge; fcluster cuts the tree at k.
Compare linkage methods on the same scaled points. Cluster IDs have no meaning.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X, _ = make_blobs(n_samples=80, centers=3, cluster_std=1., random_state=9)
X = StandardScaler().fit_transform(X)
for method in ("single", "complete", "average", "ward"):
    tree = linkage(X, method=method, metric="euclidean")
    labels = fcluster(tree, t=3, criterion="maxclust")
    sizes = [int(np.sum(labels == label)) for label in np.unique(labels)]
    print(f"{method:8s}: groups={len(sizes)}, first merge distance={tree[0,2]:.3f}, sizes={sizes}")
