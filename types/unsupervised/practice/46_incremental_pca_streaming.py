"""46. Incremental PCA: fit PCA on data that arrives in batches."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(load_digits().data)

batch = IncrementalPCA(n_components=20)
for chunk in np.array_split(X, 10):
    batch.partial_fit(chunk)
X_ipca = batch.transform(X)

full = PCA(n_components=20, random_state=42).fit(X)
X_pca = full.transform(X)

print("incremental explained variance:", round(batch.explained_variance_ratio_.sum(), 4))
print("full PCA explained variance:   ", round(full.explained_variance_ratio_.sum(), 4))
print("MSE between the two embeddings:", round(mean_squared_error(X_ipca, X_pca), 4))
