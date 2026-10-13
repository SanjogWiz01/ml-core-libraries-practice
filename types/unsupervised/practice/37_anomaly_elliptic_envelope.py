"""37. Elliptic Envelope: robust covariance for anomaly detection."""
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
normal = rng.normal(0, 1, size=(200, 2))
anomalies = rng.uniform(-8, 8, size=(10, 2))
X = np.vstack([normal, anomalies])

X_scaled = StandardScaler().fit_transform(X)

model = EllipticEnvelope(contamination=10 / 210, random_state=42).fit(X_scaled)
pred = model.predict(X_scaled)               # 1 = inlier, -1 = outlier
scores = model.decision_function(X_scaled)

flagged = np.where(pred == -1)[0]
print("flagged indices:", flagged)
print("ground-truth anomalies (last 10):", list(range(200, 210)))
print("caught:", sorted(set(flagged) & set(range(200, 210))))
print("location (robust mean):", model.location_.round(2))
