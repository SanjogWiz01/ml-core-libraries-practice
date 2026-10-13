"""43. Bayesian GMM: let the model shrink unused components itself."""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import BayesianGaussianMixture

X, _ = make_blobs(n_samples=800, centers=5, cluster_std=0.9, random_state=42)

bayes = BayesianGaussianMixture(
    n_components=10,
    covariance_type="full",
    weight_concentration_prior_type="dirichlet_process",
    random_state=42,
).fit(X)
labels = bayes.predict(X)

weights = bayes.weights_
print("component weights (near-zero = effectively unused):")
print(weights.round(3))
print("effective clusters (weight > 0.01):", int((weights > 0.01).sum()))
print("cluster counts:", np.bincount(labels))
print("we asked for 10 components but the prior pruned empty ones.")
