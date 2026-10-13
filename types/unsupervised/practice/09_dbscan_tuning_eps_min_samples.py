"""9. Tuning DBSCAN: eps and min_samples change everything."""
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

X, _ = make_moons(n_samples=500, noise=0.06, random_state=42)

for eps in (0.05, 0.1, 0.2, 0.4):
    for min_samples in (3, 5, 10):
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise = int((labels == -1).sum())
        print(f"eps={eps:<4} min_samples={min_samples:<3} -> clusters={clusters}, noise={noise}")
    print()
