# Unsupervised Learning

Unsupervised learning works with **unlabelled** data. There is no `y` to
supervise the model, so the model must find structure in `X` on its own.

The goal is usually one of three things:

- **Grouping** - similar points end up in the same cluster
- **Compression** - represent the data with fewer dimensions, keeping the signal
- **Density / anomaly structure** - find which points are unusual

## Main Problem Types

| Type | Purpose | Key algorithms |
|------|---------|----------------|
| **Clustering** | discover groups | K-Means, DBSCAN, Hierarchical, GMM |
| **Dimensionality reduction** | compress / visualise | PCA, t-SNE, UMAP, LDA, Autoencoders |
| **Association rules** | co-occurrence mining | Apriori, FP-Growth |
| **Anomaly detection** | find rare points | IsolationForest, One-ClassSVM |

## What Makes It Harder Than Supervised

- **No ground truth.** You cannot say whether cluster A is "correct". You must
  interpret them, so domain knowledge matters more.
- **No free evaluation metric.** Metrics such as silhouette score or inertia are
  proxies, not proof of quality.
- **Results are scale sensitive.** Almost every algorithm here requires
  standardisation first, because distance/variance units change everything.

## Canonical Workflow

```
1. Load + explore     -> distributions, missing values, outliers
2. Scale              -> StandardScaler / MinMaxScaler  (almost always)
3. Train the model    -> K-Means(n), PCA(k=2), DBSCAN(eps, min_samples)
4. Inspect results    -> inertia, silhouette, explained_variance_ratio
5. Visualise          -> 2-D scatter coloured by cluster / principal component
6. Use the output     -> customer segments, image compression, deduping
```

## Concrete Uses

- Customer segmentation for marketing
- Grouping documents after removing labels
- Compressing thousands of image pixels down to 50 latent features
- Fraud / anomaly flags where "normal" is the only known class

## Current Contents

Runnable practice scripts live in [`practice/`](practice/). Each file is numbered
in a suggested learning order.

| Range | Focus |
|-------|-------|
| `01`-`05` | K-Means, centroids/inertia, elbow, silhouette, scaling |
| `06`-`09` | Hierarchical clustering, dendrograms, DBSCAN |
| `10`-`13` | PCA (basics, variance, visualisation), t-SNE |
| `14`-`17` | Gaussian Mixture, Isolation Forest, validity metrics, MiniBatch |
| `18`-`21` | Image quantization, PCA reconstruction, Kernel PCA, NMF |
| `22`-`25` | One-Class SVM, LOF, Spectral clustering, text clustering |
| `26`-`30` | k-means++ init, connectivity constraints, BIRCH, Mean Shift, Affinity Propagation |
| `31`-`36` | PCA whitening, TruncatedSVD/LSA, FastICA, Factor Analysis, LDA, Isomap |
| `37`-`38` | Elliptic Envelope, cluster profiling |
| `39`-`41` | LLE, MDS, Spectral embedding |
| `42`-`45` | GMM model selection, Bayesian GMM, stability, OPTICS |
| `46`-`48` | Incremental PCA, feature agglomeration, Apriori association rules |
| `49`-`51` | PCA reconstruction anomalies, anomaly ensemble, silhouette diagram |

## Related

- [`../supervised/`](../supervised/)
- [`../reinforcement-learning/`](../reinforcement-learning/)
