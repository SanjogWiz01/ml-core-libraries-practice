"""36. Isomap: nonlinear manifold learning using geodesic distances."""
from sklearn.datasets import load_digits
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X = StandardScaler().fit_transform(digits.data)

isomap = Isomap(n_neighbors=10, n_components=2)
embedding = isomap.fit_transform(X)

print("input shape: ", X.shape)
print("embedding shape:", embedding.shape)
print("reconstruction error:", round(isomap.reconstruction_error(), 4))
print("first 3 embedded points:")
print(embedding[:3].round(3))
