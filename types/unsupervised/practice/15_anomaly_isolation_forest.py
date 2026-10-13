"""15. Anomaly detection with Isolation Forest (no labels needed)."""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
normal = rng.normal(0, 1, size=(490, 2))
anomalies = rng.uniform(-8, 8, size=(10, 2))
X = np.vstack([normal, anomalies])

X_scaled = StandardScaler().fit_transform(X)

iso = IsolationForest(contamination=10 / 500, random_state=42)
pred = iso.fit_predict(X_scaled)          # 1 = inlier, -1 = outlier
scores = iso.decision_function(X_scaled)  # lower = more anomalous

flagged = np.where(pred == -1)[0]
print("flagged indices:", flagged)
print("ground-truth anomaly indices (last 10):", list(range(490, 500)))
print("actually caught:", sorted(set(flagged) & set(range(490, 500))))
print("lowest 5 anomaly scores:", np.sort(scores)[:5].round(3))
