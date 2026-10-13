"""50. Anomaly ensemble: combine several detectors by majority vote."""
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

rng = np.random.default_rng(42)
normal = rng.normal(0, 1, size=(300, 2))
anomalies = rng.uniform(-7, 7, size=(10, 2))
X = StandardScaler().fit_transform(np.vstack([normal, anomalies]))

detectors = {
    "isolation_forest": IsolationForest(contamination=0.03, random_state=42),
    "one_class_svm": OneClassSVM(nu=0.03),
    "elliptic_envelope": EllipticEnvelope(contamination=0.03, random_state=42),
    "local_outlier_factor": LocalOutlierFactor(n_neighbors=20, contamination=0.03),
}

votes = np.zeros(len(X))
for name, detector in detectors.items():
    pred = detector.fit_predict(X)          # -1 = outlier
    votes += (pred == -1).astype(int)
    print(f"{name:<21} flagged {int((pred == -1).sum())}")

ensemble = np.where(votes >= 3)[0]
print("majority-vote ensemble flagged:", ensemble)
print("ground-truth anomalies (last 10):", list(range(300, 310)))
print("caught:", sorted(set(ensemble) & set(range(300, 310))))
