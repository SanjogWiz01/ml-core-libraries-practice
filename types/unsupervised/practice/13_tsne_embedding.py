"""13. t-SNE: nonlinear embedding that respects local neighbourhoods."""
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X = StandardScaler().fit_transform(digits.data)

embedding = TSNE(n_components=2, perplexity=30, random_state=42, init="pca").fit_transform(X)

plt.figure(figsize=(7, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap="tab10", s=8)
plt.title("t-SNE of digits (colour = true digit, for reference only)")
plt.show()

print("embedding shape:", embedding.shape)
