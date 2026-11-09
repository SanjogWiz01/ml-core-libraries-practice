"""ADVANCED RESEARCH PAPERS — a guided map and runnable model comparison.

Read methods papers by asking: (1) what objective and assumptions are introduced,
(2) what data and baselines were used, (3) which claims are theoretical versus
empirical, and (4) what later work says is fragile? Paper ideas are starting
points; implementation details and benchmarks evolve.

Selected primary papers:
* Lloyd (1982), least-squares quantization / centroid assignment:
  https://doi.org/10.1109/TIT.1982.1056489
* Ester et al. (1996), DBSCAN density reachability and noise:
  https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf
* Dempster, Laird & Rubin (1977), EM algorithms and likelihood inference:
  https://doi.org/10.1111/j.2517-6161.1977.tb01600.x
* Rousseeuw (1987), silhouette interpretation and validation:
  https://doi.org/10.1016/0377-0427(87)90125-7
* Liu, Ting & Zhou (2008), Isolation Forest anomaly detection:
  https://doi.org/10.1109/ICDM.2008.17
* van der Maaten & Hinton (2008), t-SNE neighborhood embedding:
  https://www.jmlr.org/papers/v9/vandermaaten08a.html
* Hotelling (1933), principal component analysis foundations:
  https://doi.org/10.1093/biomet/25.3-4.321
* Blei, Ng & Jordan (2003), LDA generative topic model:
  https://jmlr.org/papers/v3/blei03a.html
* Hinton & Salakhutdinov (2006), neural dimensionality reduction:
  https://www.science.org/doi/10.1126/science.1127647
* McInnes, Healy & Melville (2018), UMAP manifold embedding:
  https://arxiv.org/abs/1802.03426

IMPLEMENTATION: concentric circles violate K-Means' spherical cluster assumption.
K-Means, DBSCAN, and a two-component GMM use identical points. Synthetic labels
are used after fitting only for teaching comparison; real unlabeled data has none.
"""
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_circles
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

X, demonstration_truth = make_circles(n_samples=500, factor=.45, noise=.04, random_state=27)
X = StandardScaler().fit_transform(X)
predictions = {
    "K-Means": KMeans(n_clusters=2, n_init=20, random_state=27).fit_predict(X),
    "DBSCAN": DBSCAN(eps=.22, min_samples=6).fit_predict(X),
    "GMM": GaussianMixture(n_components=2, n_init=5, random_state=27).fit_predict(X),
}
for name, labels in predictions.items():
    print(f"{name:8s} clusters={len(set(labels)-{-1})}, noise={(labels == -1).sum()}, "
          f"ARI_vs_synthetic_truth={adjusted_rand_score(demonstration_truth, labels):.3f}")
