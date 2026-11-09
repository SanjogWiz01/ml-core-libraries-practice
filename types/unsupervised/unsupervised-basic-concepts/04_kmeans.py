"""K-MEANS — partition observations by nearest centroid.

K-Means alternates assigning each point to its nearest centroid and replacing
each centroid with its members' mean. It minimizes within-cluster squared
Euclidean distance (inertia), a non-convex objective. It is fast but favors
roughly spherical, similarly sized groups and requires k. k-means++ and multiple
restarts reduce poor starts but do not guarantee the global optimum.

IMPLEMENTATION: make_blobs provides unlabeled observations for fitting; helper
truth is used only after fitting to demonstrate optional external evaluation.
Inspect centers_, labels_, and inertia_ below.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

X, illustrative_truth = make_blobs(n_samples=300, centers=4, cluster_std=.75, random_state=12)
model = KMeans(n_clusters=4, init="k-means++", n_init=20, random_state=12)
labels = model.fit_predict(X)  # truth is not passed to the model
print("Centroids:\n", np.round(model.cluster_centers_, 2))
print("Cluster sizes:", np.bincount(labels).tolist())
print("Inertia:", round(model.inertia_, 2))
print("ARI vs toy generator labels:", round(adjusted_rand_score(illustrative_truth, labels), 3))
