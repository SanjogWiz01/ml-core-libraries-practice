"""31. PCA whitening: decorrelate components and scale them to unit variance."""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

X = load_iris().data
X_centered = X - X.mean(axis=0)

plain = PCA(whiten=False, random_state=42).fit_transform(X_centered)
white = PCA(whiten=True, random_state=42).fit_transform(X_centered)

print("plain PCA component covariance (diagonal):",
      np.round(np.var(plain, axis=0), 3))
print("whitened covariance (off-diagonals ~0, diagonal ~1):")
print(np.cov(white, rowvar=False).round(3))
