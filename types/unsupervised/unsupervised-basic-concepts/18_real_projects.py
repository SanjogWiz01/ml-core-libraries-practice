"""REAL PROJECTS — end-to-end synthetic customer segmentation exercise.

Define the use case; validate data; select features available at decision time;
respect time/customer boundaries; fit preprocessing on training rows only;
compare plausible models; profile groups in original units; check stability,
privacy, fairness, and downstream value; then pilot and monitor. Clusters are
not customer truths and should not automatically determine consequential access.

IMPLEMENTATION: generated transaction aggregates mimic a feature table; a
Pipeline scales and clusters consistently; groupby profiles in raw units; a
resampled fit provides a rough stability check. Hidden generator groups are
used only to create synthetic structure and are never supplied to the model.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

rng = np.random.default_rng(202)
n = 240
generator_group = rng.choice(3, size=n, p=[.4, .35, .25])
means = np.array([[3, 45, 1], [9, 220, 4], [5, 110, 10]])
values = means[generator_group] + rng.normal(0, [1.5, 35, 2], size=(n, 3))
customers = pd.DataFrame(values, columns=["visits_month", "spend_month", "support_tickets"])
customers["spend_month"] = customers["spend_month"].clip(lower=0)
features = customers.columns.tolist()
pipeline = make_pipeline(StandardScaler(), KMeans(n_clusters=3, n_init=20, random_state=202))
clusters = pipeline.fit_predict(customers[features])
customers["cluster"] = clusters
print("Cluster profile (means in original units):\n", customers.groupby("cluster")[features].mean().round(1))
print("Cluster sizes:", customers["cluster"].value_counts().sort_index().to_dict())
resampled = customers[features].sample(frac=.85, replace=True, random_state=5)
second = make_pipeline(StandardScaler(), KMeans(n_clusters=3, n_init=20, random_state=5))
second_labels = second.fit_predict(resampled)
print("ARI between bootstrap fit and corresponding original labels:",
      round(adjusted_rand_score(clusters[resampled.index.to_numpy()], second_labels), 3))
print("Next: validate profiles with domain experts and a controlled business pilot.")
