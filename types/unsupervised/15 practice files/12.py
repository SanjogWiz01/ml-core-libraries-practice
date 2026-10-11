"""12. PCA for 2-D visualisation of high-dimensional data."""
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(load_iris().data)

X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=load_iris().target, cmap="viridis", alpha=0.8)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Iris projected onto its first two principal components")
plt.show()
