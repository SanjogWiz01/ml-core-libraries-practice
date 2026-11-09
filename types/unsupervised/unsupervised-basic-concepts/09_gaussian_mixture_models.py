"""GAUSSIAN MIXTURE MODELS — probabilistic clusters with soft memberships.

A GMM models data as a weighted sum of Gaussian components. Expectation-
Maximization alternates estimating component responsibilities and updating
weights, means, and covariance. predict_proba expresses membership uncertainty;
covariance_type controls shape assumptions. Choose component count carefully:
AIC/BIC trade fit against complexity but do not guarantee useful segments.
Gaussian assumptions and nearly singular components need attention.

IMPLEMENTATION: fit a full-covariance model and inspect probabilities. Then
compare BIC/AIC for candidate component counts; lower is preferred within this
model family and dataset, not an absolute quality score.
"""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

X, _ = make_blobs(n_samples=500, centers=3, cluster_std=[.55, 1., 1.4], random_state=32)
X = StandardScaler().fit_transform(X)
model = GaussianMixture(n_components=3, covariance_type="full", n_init=5, random_state=32).fit(X)
print("First five component probabilities:\n", np.round(model.predict_proba(X[:5]), 3))
for k in range(1, 7):
    candidate = GaussianMixture(n_components=k, n_init=3, random_state=32).fit(X)
    print(f"k={k}: BIC={candidate.bic(X):.1f}, AIC={candidate.aic(X):.1f}")
