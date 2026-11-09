"""ANOMALY DETECTION — rank observations that differ from learned normality.

Anomaly is task-relative. Isolation Forest partitions features; unusual rows
tend to be isolated by shorter paths. decision_function is higher for inliers
and lower for outliers; predict returns 1 for inliers and -1 for outliers.
contamination influences the threshold and is not a prevalence fact. Investigate
alerts with context and feedback before consequential automation.

IMPLEMENTATION: fit mostly normal generated observations; score all rows; print
the most suspicious alongside known injected anomalies for this demo only.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(10)
normal = rng.normal(loc=[0, 0], scale=[1, .7], size=(300, 2))
anomalies = rng.uniform(low=-6, high=6, size=(12, 2))
X = StandardScaler().fit_transform(np.vstack([normal, anomalies]))
model = IsolationForest(n_estimators=200, contamination=.04, random_state=10)
model.fit(X[:300])  # illustrative mostly-normal training set
scores, predictions = model.decision_function(X), model.predict(X)
print("Predicted outlier count:", int(np.sum(predictions == -1)))
print("Most suspicious rows (index, score, injected anomaly?):")
for i in np.argsort(scores)[:10]:
    print(int(i), round(float(scores[i]), 3), bool(i >= len(normal)))
