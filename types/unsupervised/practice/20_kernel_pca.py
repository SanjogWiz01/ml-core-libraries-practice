"""20. Kernel PCA: linear PCA cannot unfold, a kernel can."""
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA, KernelPCA

X, y = make_circles(n_samples=500, factor=0.3, noise=0.05, random_state=42)

linear = PCA(n_components=2, random_state=42).fit_transform(X)
rbf = KernelPCA(n_components=2, kernel="rbf", gamma=15, random_state=42).fit_transform(X)

print("linear PCA  -> first component range:", round(linear[:, 0].min(), 2), "to", round(linear[:, 0].max(), 2))
print("kernel PCA  -> first component range:", round(rbf[:, 0].min(), 2), "to", round(rbf[:, 0].max(), 2))
print("RBF kernel makes the inner and outer ring linearly separable.")
