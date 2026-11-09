"""PCA — rotate centered data into orthogonal directions of decreasing variance.

Principal Component Analysis is a linear projection. With centered X, it finds
directions maximizing variance; projecting to fewer components compresses data
and may remove noise, but can discard useful low-variance signal. Scale first
when units differ. Component signs are arbitrary; components are mathematical
combinations, not causal factors.

IMPLEMENTATION: standardize features, fit two-component PCA, project with
transform, then reconstruct with inverse_transform. Explained variance and MSE
summarize what the selected projection retains and loses.
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(load_iris().data)
pca = PCA(n_components=2, random_state=0)
Z = pca.fit_transform(X)
reconstructed = pca.inverse_transform(Z)
print("Original/projected shape:", X.shape, Z.shape)
print("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 3))
print("Cumulative variance retained:", round(pca.explained_variance_ratio_.sum(), 3))
print("2-component reconstruction MSE:", round(float(np.mean((X-reconstructed)**2)), 3))
print("Components (rows are directions):\n", np.round(pca.components_, 3))
