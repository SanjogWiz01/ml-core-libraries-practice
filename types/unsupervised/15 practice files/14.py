"""14. Gaussian Mixture Model: soft, probabilistic clustering."""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)

gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=42)
labels = gmm.fit_predict(X)
probs = gmm.predict_proba(X)

print("cluster counts:", np.bincount(labels))
print("silhouette:", round(silhouette_score(X, labels), 3))
print("BIC (lower is better):", round(gmm.bic(X), 1))
print("soft responsibility for first 3 points:")
print(probs[:3].round(3))
