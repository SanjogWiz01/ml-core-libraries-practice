"""49. PCA-based anomaly detection using reconstruction error."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
normal = rng.normal(0, 1, size=(490, 6))
anomalies = rng.uniform(-6, 6, size=(10, 6))
X = np.vstack([normal, anomalies])

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=3, random_state=42).fit(X_scaled[:490])
reconstructed = pca.inverse_transform(pca.transform(X_scaled))
errors = np.sum((X_scaled - reconstructed) ** 2, axis=1)

threshold = np.percentile(errors[:490], 99)
flagged = np.where(errors > threshold)[0]
print("threshold:", round(float(threshold), 3))
print("flagged indices:", flagged)
print("ground-truth anomalies (last 10):", list(range(490, 500)))
print("caught:", sorted(set(flagged) & set(range(490, 500))))
print("points the low-rank subspace cannot reconstruct are the anomalies.")
