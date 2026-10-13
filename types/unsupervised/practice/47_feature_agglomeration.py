"""47. Feature agglomeration: merge similar features before clustering."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(load_digits().data)

agglo = FeatureAgglomeration(n_clusters=16, linkage="ward")
X_reduced = agglo.fit_transform(X)

print("original features:", X.shape[1])
print("reduced features: ", X_reduced.shape[1])
print("group sizes:", agglo.labels_[:16])
reconstruction = agglo.inverse_transform(X_reduced)
print("reconstruction MSE:", round(np.mean((X - reconstruction) ** 2), 4))
print("correlated neighbouring pixels are merged into one super-feature.")
