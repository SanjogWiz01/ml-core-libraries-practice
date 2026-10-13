"""39. Locally Linear Embedding: nonlinear manifold unfolding."""
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

X, color = make_swiss_roll(n_samples=1000, noise=0.05, random_state=42)

for n_neighbors in (5, 12, 30):
    lle = LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=2, random_state=42
    ).fit(X)
    print(f"n_neighbors={n_neighbors:<3} reconstruction_error={lle.reconstruction_error_:.4f}")

lle = LocallyLinearEmbedding(n_neighbors=12, n_components=2, random_state=42)
embedding = lle.fit_transform(X)

plt.figure(figsize=(7, 5))
plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=8)
plt.title("LLE of the swiss roll")
plt.xlabel("component 1")
plt.ylabel("component 2")
plt.show()
