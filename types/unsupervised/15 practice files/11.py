"""11. PCA explained variance: how many components do we need?"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(load_iris().data)

pca = PCA(random_state=42).fit(X)
ratio = pca.explained_variance_ratio_
cumulative = np.cumsum(ratio)

for i, (r, c) in enumerate(zip(ratio, cumulative), start=1):
    print(f"PC{i}: {r:.3f}  cumulative: {c:.3f}")

n_needed = int(np.searchsorted(cumulative, 0.95) + 1)
print("components to keep 95% of variance:", n_needed)
