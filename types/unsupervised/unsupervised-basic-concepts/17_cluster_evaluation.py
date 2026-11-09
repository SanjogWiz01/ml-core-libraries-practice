"""CLUSTER EVALUATION — ask whether discovered structure is useful and robust.

Internal scores (silhouette, Davies-Bouldin, Calinski-Harabasz) use geometry
and have assumptions. External scores compare with reference labels when they
exist, but those labels must not leak into fitting. Stability checks whether
groups persist under resampling, perturbation, or reasonable parameter changes.
Also profile clusters in original units, inspect size and interpretability, and
measure downstream utility. No single score certifies a clustering.

IMPLEMENTATION: compare two methods on identical data using three internal
metrics. Generator labels are used after fitting only as an external sanity check.
"""
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)
from sklearn.preprocessing import StandardScaler

X, reference = make_blobs(n_samples=300, centers=4, cluster_std=1., random_state=13)
X = StandardScaler().fit_transform(X)
models = {"kmeans": KMeans(n_clusters=4, n_init=15, random_state=13),
          "ward": AgglomerativeClustering(n_clusters=4, linkage="ward")}
for name, model in models.items():
    labels = model.fit_predict(X)
    print(name, "silhouette=", round(silhouette_score(X, labels), 3),
          "Davies-Bouldin=", round(davies_bouldin_score(X, labels), 3),
          "Calinski-Harabasz=", round(calinski_harabasz_score(X, labels), 1),
          "ARI (toy truth)=", round(adjusted_rand_score(reference, labels), 3))
