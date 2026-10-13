"""23. Local Outlier Factor: density-based anomaly detection."""
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

rng = np.random.default_rng(42)
X_train = rng.normal(0, 1, size=(300, 2))
X_test = np.vstack([rng.normal(0, 1, (95, 2)),
                    rng.uniform(-6, 6, (5, 2))])

model = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
train_labels = model.fit_predict(X_train)            # novelty=False mode
print("outliers found in training normal data:", int((train_labels == -1).sum()))

novelty = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(X_train)
pred = novelty.predict(X_test)
flagged = np.where(pred == -1)[0]
print("flagged in test:", flagged)
print("caught true anomalies:", sorted(set(flagged) & set(range(95, 100))))
print("negative LOF scores (lower = more anomalous):",
      np.sort(novelty.decision_function(X_test))[:5].round(3))
