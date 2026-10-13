"""41. Spectral Embedding: Laplacian eigenmaps for nonlinear structure."""
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.manifold import SpectralEmbedding

X, y = make_moons(n_samples=500, noise=0.06, random_state=42)

embedding = SpectralEmbedding(
    n_components=2, affinity="nearest_neighbors", n_neighbors=10, random_state=42
).fit_transform(X)

print("embedding shape:", embedding.shape)
print("first 3 embedded points:")
print(embedding[:3].round(3))

plt.figure(figsize=(7, 5))
plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap="coolwarm", s=12)
plt.title("Spectral embedding of two moons")
plt.show()
