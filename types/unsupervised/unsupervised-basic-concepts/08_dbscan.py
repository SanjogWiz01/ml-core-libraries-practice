"""DBSCAN — discover dense connected regions and mark sparse points as noise.

DBSCAN uses eps (neighborhood radius) and min_samples (density threshold). Core
points have enough nearby observations; border points are near a core but do
not meet the threshold; others receive label -1. It needs no k and can find
non-convex shapes. One eps struggles with varying densities. Scaling and metric
matter; inspect k-distance plots and choose a radius meaningful for the domain.

IMPLEMENTATION: scale curved moon data and fit multiple eps choices. Count
non-noise groups and noise separately so parameter effects are clear.
"""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

X, _ = make_moons(n_samples=350, noise=.07, random_state=4)
X = StandardScaler().fit_transform(X)
for eps in (.12, .18, .25, .35):
    labels = DBSCAN(eps=eps, min_samples=6).fit_predict(X)
    groups = sorted(set(labels) - {-1})
    print(f"eps={eps:.2f}: clusters={len(groups)}, noise={int(np.sum(labels == -1))}, "
          f"sizes={[int(np.sum(labels == group)) for group in groups]}")
