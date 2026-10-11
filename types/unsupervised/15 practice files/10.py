"""10. PCA basics: principal components as new axes."""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(load_iris().data)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

print("input shape: ", X.shape)
print("output shape:", X_pca.shape)
print("components (rows = new axes):")
print(pca.components_.round(3))
print("first 3 projected points:")
print(X_pca[:3].round(3))
