# Customer Segmentation

**Type**: Unsupervised Clustering  
**Dataset**: Synthetic (800 customers) — `data/raw/customer_segmentation.csv`  
**Target**: None (find clusters)

## Skills Demonstrated
- `StandardScaler` before clustering (mandatory)
- K-Means with elbow method + silhouette score to find K
- DBSCAN with eps sensitivity analysis
- PCA to 2D for cluster visualization
- Cluster profiling: interpret what each segment means

## Run
```bash
python main.py
```
