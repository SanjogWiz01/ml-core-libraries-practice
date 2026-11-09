"""MODEL DEPLOYMENT — save preprocessing and estimator as one artifact.

For a distance-based model, saving only the clusterer loses the scaler and can
silently change assignments. Persist feature order, imputation, scaling, and
model together in a Pipeline. Load only trusted joblib/pickle artifacts: loading
can execute arbitrary code. Record library versions, configuration, data
lineage, and evaluation. Monitor input drift, cluster-size shifts, and outcomes.

IMPLEMENTATION: fit a pipeline, save it in a temporary directory, reload it,
assign new observations, then clean up the temporary artifact.
"""
from pathlib import Path
import tempfile
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, _ = make_blobs(n_samples=180, centers=3, cluster_std=.9, random_state=18)
pipeline = make_pipeline(StandardScaler(), KMeans(n_clusters=3, n_init=15, random_state=18))
pipeline.fit(X)
with tempfile.TemporaryDirectory() as directory:
    artifact = Path(directory) / "customer_clusters.joblib"
    joblib.dump(pipeline, artifact)
    loaded = joblib.load(artifact)  # only load artifacts you created or trust
    sample = np.array([[0., 0.], [4., 4.]])
    print("Artifact bytes:", artifact.stat().st_size)
    print("New row cluster IDs:", loaded.predict(sample).tolist())
    print("These examples expect two ordered numeric columns; production needs a schema.")
