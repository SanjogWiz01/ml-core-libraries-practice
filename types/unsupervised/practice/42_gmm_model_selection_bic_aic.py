"""42. GMM model selection: choose components with BIC / AIC."""
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

X, _ = make_blobs(n_samples=800, centers=5, cluster_std=0.9, random_state=42)

results = []
for k in range(1, 11):
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42).fit(X)
    results.append((k, gmm.bic(X), gmm.aic(X)))

for k, bic, aic in results:
    print(f"k={k:<3} BIC={bic:>10.1f}  AIC={aic:>10.1f}")

best_bic = min(results, key=lambda r: r[1])[0]
best_aic = min(results, key=lambda r: r[2])[0]
print("best k by BIC:", best_bic, " best k by AIC:", best_aic)
print("lower BIC/AIC is better; BIC penalises complexity harder.")
