"""22. One-Class SVM: learn the boundary of 'normal' from normal data only."""
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
X_train = rng.normal(0, 1, size=(300, 2))            # only normal points seen
X_test = np.vstack([rng.normal(0, 1, (95, 2)),       # normal
                    rng.uniform(-6, 6, (5, 2))])     # anomalies

scaler = StandardScaler().fit(X_train)
model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.02).fit(scaler.transform(X_train))

pred = model.predict(scaler.transform(X_test))       # 1 = normal, -1 = anomaly
flagged = np.where(pred == -1)[0]
print("flagged indices:", flagged)
print("ground-truth anomalies (last 5):", list(range(95, 100)))
print("caught:", sorted(set(flagged) & set(range(95, 100))))
