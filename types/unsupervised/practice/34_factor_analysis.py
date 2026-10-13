"""34. Factor Analysis: a generative latent-variable model vs PCA."""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(load_iris().data)

fa = FactorAnalysis(n_components=2, random_state=42)
X_fa = fa.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

print("Factor Analysis - mixing matrix (loadings):")
print(fa.components_.round(3))
print("unique variances per feature (noise):", fa.noise_variance_.round(3))
print()
print("PCA - loadings:")
print(pca.components_.round(3))
print()
print("FA models feature-specific noise; PCA assumes isotropic noise.")
