"""40. Multidimensional Scaling: preserve pairwise distances in 2-D."""
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X = StandardScaler().fit_transform(digits.data[:400])

mds = MDS(
    n_components=2,
    metric="euclidean",
    init="random",
    n_init=1,
    max_iter=300,
    random_state=42,
)
embedding = mds.fit_transform(X)

print("embedding shape:", embedding.shape)
print("normalised stress:", round(mds.stress_, 1))

plt.figure(figsize=(7, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target[:400], cmap="tab10", s=12)
plt.title("MDS of 400 digits")
plt.show()
