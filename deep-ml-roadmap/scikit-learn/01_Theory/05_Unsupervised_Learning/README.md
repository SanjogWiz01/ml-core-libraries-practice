# Unsupervised Learning

No labels. The algorithm finds structure in the data on its own.

Two main tasks covered here: **Clustering** and **Dimensionality Reduction**.

---

## CLUSTERING

### K-Means

**What**: Partitions data into K clusters by minimizing the within-cluster sum of squared distances to cluster centers (centroids).

**Algorithm**:
```
1. Initialize K centroids randomly
2. Assign each point to nearest centroid
3. Recompute centroid as mean of assigned points
4. Repeat 2-3 until centroids stop moving
```

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_scaled)
labels = kmeans.labels_           # cluster assignment per sample
centers = kmeans.cluster_centers_ # centroid locations
inertia = kmeans.inertia_         # sum of squared distances to nearest centroid
```

**When to use**:
- You want compact, spherical clusters
- K is known or discoverable via elbow method
- Large datasets (linear complexity)

**Advantages**: Fast, scalable, interpretable centroids
**Disadvantages**: Must specify K, assumes spherical clusters, sensitive to outliers and scale, non-deterministic without fixing seed

**Key hyperparameters**:
- `n_clusters` — K (most important — use elbow method or silhouette score)
- `init` — centroid initialization ('k-means++' default, smarter than random)
- `n_init` — number of random initializations (use best result; default 10)

### Finding K — Elbow Method

```python
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Plot: look for the "elbow" where inertia stops dropping sharply
plt.plot(range(1, 11), inertias, 'bo-')
plt.xlabel('K')
plt.ylabel('Inertia')
```

The elbow is the point of diminishing returns — more clusters beyond it don't buy much.

---

### DBSCAN

**What**: Density-Based Spatial Clustering. Groups points that are close together (high density), marks outliers as noise (-1).

**Concept**:
```
Core point: has ≥ min_samples neighbors within radius ε
Border point: within ε of a core point, but fewer than min_samples neighbors
Noise point: not a core or border point → labeled -1
```

```python
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.5, min_samples=5)
db.fit(X_scaled)
labels = db.labels_    # -1 = noise, 0,1,2,... = clusters
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
```

**When to use**:
- You don't know K
- Clusters have irregular shapes (not spherical)
- You want outlier detection as a by-product
- Points can be noise (don't all need to belong to a cluster)

**Advantages**: No K needed, finds arbitrary shapes, identifies outliers, not sensitive to initialization
**Disadvantages**: Two hyperparameters (ε and min_samples) that are hard to set, struggles with varying density clusters, slow on very large data

**Key hyperparameters**:
- `eps` — neighborhood radius (use k-distance plot: sort distances to k-th neighbor, find elbow)
- `min_samples` — min points to form a core (rule of thumb: 2 × n_features, minimum 3)

---

## CLUSTERING EVALUATION

### Challenge
Without labels, we can't use standard classification metrics.

### Inertia / WCSS (KMeans only)
Sum of squared distances from each point to its cluster centroid.
- Lower is better
- Use for elbow method — not for comparing K-Means to other algorithms

### Silhouette Score

For each sample: `s = (b - a) / max(a, b)`

Where:
- `a` = mean distance to other samples in **same** cluster (cohesion)
- `b` = mean distance to samples in **nearest other** cluster (separation)

Range: [-1, 1]
- 1.0 = perfectly clustered
- 0.0 = on the boundary between clusters
- -1.0 = wrong cluster

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X_scaled, labels)
print(f"Silhouette score: {score:.3f}")
```

Higher is better. Use to choose K or compare clustering algorithms.

---

## DIMENSIONALITY REDUCTION — PCA

### What Is PCA?

Principal Component Analysis projects data onto fewer dimensions that capture maximum variance.

```
Original data: 50 features (many correlated)
                ↓ PCA
Compressed data: 2-10 principal components
```

**What it does**:
1. Centers the data (subtract mean)
2. Computes the directions (principal components) that have the most variance
3. Projects data onto those components

**Principal components**:
- PC1 = direction of maximum variance in the data
- PC2 = direction of maximum remaining variance, orthogonal to PC1
- PC3 = next most variance, orthogonal to PC1 and PC2
- ...

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)                    # keep 2 components
X_pca = pca.fit_transform(X_scaled)         # fit on train, transform

explained = pca.explained_variance_ratio_    # [0.45, 0.30] means 75% total
print(f"Explained variance: {explained.cumsum()}")
```

### Choosing n_components — Explained Variance

```python
pca = PCA()              # keep all components
pca.fit(X_scaled)
cumulative_variance = pca.explained_variance_ratio_.cumsum()

# Find n_components for 95% explained variance
n_95 = (cumulative_variance < 0.95).sum() + 1
print(f"{n_95} components explain 95% of variance")
```

Scree plot: plot `explained_variance_ratio_` vs component number. Look for the elbow.

### When to use PCA

**Good uses**:
- Visualization: reduce to 2D/3D to see cluster structure
- Speed: reduce features before feeding to slow models
- Noise reduction: remove low-variance components that may be noise
- Multicollinearity: PCA components are uncorrelated

**Don't use PCA when**:
- Interpretability matters — components are linear combinations, not original features
- Features are already low-dimensional (e.g., 5 features)
- You need original feature names for business explanation

### PCA vs Feature Selection

| | PCA | Feature Selection |
|---|---|---|
| Creates new features? | Yes (combinations) | No (selects original) |
| Interpretable? | No | Yes |
| Removes noise? | Yes | Depends |
| Handles correlations? | Yes | No |

### Full Pipeline with PCA

```python
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca',    PCA(n_components=10)),
    ('svm',    SVC())
])
pipe.fit(X_train, y_train)
```

Always scale before PCA — PCA is variance-based, so scale differences dominate without scaling.
