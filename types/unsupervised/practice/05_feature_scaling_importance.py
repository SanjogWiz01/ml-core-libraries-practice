"""5. Why scaling matters: distance-based models are unit sensitive."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
X = np.column_stack([
    rng.normal(50_000, 10_000, 500),  # income, huge scale
    rng.normal(30, 5, 500),           # age, small scale
])

raw = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X)
scaled = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(
    StandardScaler().fit_transform(X)
)

print("cluster sizes on raw data:   ", np.bincount(raw))
print("cluster sizes on scaled data:", np.bincount(scaled))
print("income dominates the raw distances -> always scale first.")
