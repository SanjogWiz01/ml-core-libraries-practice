# Unsupervised Learning: Basic Concepts

This folder introduces methods for finding structure when no target labels are supplied. Each numbered Python file is one lesson: it explains the idea, creates a small reproducible dataset, implements the method, and prints results. Run any lesson from this folder or the repository root.

## Setup

Use Python 3.10+ in a virtual environment. Core lessons use NumPy, SciPy, pandas, and scikit-learn:

```bash
python -m pip install numpy scipy pandas scikit-learn
python 01_data_preprocessing.py
```

UMAP is optional:

```bash
python -m pip install umap-learn
python 12_umap.py
```

Without UMAP installed, its lesson prints the install command and exits cleanly. All lessons generate their own data; nothing is downloaded. Random seeds help reproducibility, though library versions and parallel execution can still change stochastic results.

## Lessons

| File | Concept | Focus |
|---|---|---|
| [01_data_preprocessing.py](01_data_preprocessing.py) | Data preprocessing | Imputation, encoding, pipelines, leakage prevention |
| [02_feature_scaling.py](02_feature_scaling.py) | Feature scaling | Standard, min-max, robust scaling and outliers |
| [03_distance_metrics.py](03_distance_metrics.py) | Distance metrics | Euclidean, Manhattan, cosine, scale effects |
| [04_kmeans.py](04_kmeans.py) | K-Means | Assignment, centroids, inertia, limitations |
| [05_elbow_method.py](05_elbow_method.py) | Elbow method | Inertia over candidate k values |
| [06_silhouette_score.py](06_silhouette_score.py) | Silhouette score | Cohesion, separation, per-cluster interpretation |
| [07_hierarchical_clustering.py](07_hierarchical_clustering.py) | Hierarchical clustering | Linkage and cutting a merge tree |
| [08_dbscan.py](08_dbscan.py) | DBSCAN | Density-connected clusters and noise |
| [09_gaussian_mixture_models.py](09_gaussian_mixture_models.py) | GMM | Soft membership, covariance, AIC/BIC |
| [10_pca.py](10_pca.py) | PCA | Variance directions, compression, reconstruction |
| [11_tsne.py](11_tsne.py) | t-SNE | Local neighborhood visualization |
| [12_umap.py](12_umap.py) | UMAP | Neighborhood graph embedding and parameters |
| [13_anomaly_detection.py](13_anomaly_detection.py) | Anomaly detection | Isolation Forest ranking and thresholding |
| [14_association_rules.py](14_association_rules.py) | Association rules | Support, confidence, and lift |
| [15_topic_modeling.py](15_topic_modeling.py) | Topic modeling | LDA topic-word and document-topic distributions |
| [16_autoencoders.py](16_autoencoders.py) | Autoencoders | Bottleneck representation learned by reconstruction |
| [17_cluster_evaluation.py](17_cluster_evaluation.py) | Cluster evaluation | Internal, external, and stability thinking |
| [18_real_projects.py](18_real_projects.py) | Real projects | Customer segmentation from data to pilot |
| [19_model_deployment.py](19_model_deployment.py) | Model deployment | Persist preprocessing and estimator together |
| [20_advanced_research_papers.py](20_advanced_research_papers.py) | Advanced research papers | Landmark work and a runnable model comparison |

Read each module docstring first, then follow comments marked `IMPLEMENTATION:` to see exactly where each concept is coded. Change one parameter at a time and observe how output changes.

## Good habits

- Treat unsupervised outputs as hypotheses. Cluster IDs have no inherent meaning.
- Scale when distance or variance makes feature units matter; do not scale mechanically without considering feature meaning.
- Fit imputers, encoders, and scalers on training data only. A pipeline keeps inference consistent and prevents leakage.
- t-SNE and UMAP emphasize neighborhoods and distort some global relationships. A visual island is not proof of a natural cluster.
- Silhouette favors compact separated groups and may mislead on density-based or non-convex structures. Check stability, usefulness, and domain interpretation too.
- Review privacy, fairness, and operational consequences before acting on discovered groups or anomaly scores.

## Selected references

- Scikit-learn documentation: [preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html), [clustering](https://scikit-learn.org/stable/modules/clustering.html), [decomposition](https://scikit-learn.org/stable/modules/decomposition.html), [mixture models](https://scikit-learn.org/stable/modules/mixture.html), [outlier detection](https://scikit-learn.org/stable/modules/outlier_detection.html), [model persistence](https://scikit-learn.org/stable/model_persistence.html).
- Rousseeuw (1987), [Silhouettes](https://doi.org/10.1016/0377-0427(87)90125-7); Ester et al. (1996), [DBSCAN](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf).
- van der Maaten & Hinton (2008), [t-SNE](https://www.jmlr.org/papers/v9/vandermaaten08a.html); McInnes et al. (2018), [UMAP](https://arxiv.org/abs/1802.03426).
- Blei, Ng & Jordan (2003), [Latent Dirichlet Allocation](https://jmlr.org/papers/v3/blei03a.html); Hinton & Salakhutdinov (2006), [neural network dimensionality reduction](https://www.science.org/doi/10.1126/science.1127647).
- Lloyd (1982), [least-squares quantization](https://doi.org/10.1109/TIT.1982.1056489); MacQueen (1967), [classification and analysis of multivariate observations](https://projecteuclid.org/euclid.bsmsp/1200512992).
