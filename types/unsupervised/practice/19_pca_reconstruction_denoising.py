"""19. PCA reconstruction error and denoising by dropping components."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

X = StandardScaler().fit_transform(load_digits().data)

for n in (5, 10, 20, 40):
    pca = PCA(n_components=n, random_state=42)
    X_red = pca.fit_transform(X)
    X_rec = pca.inverse_transform(X_red)
    print(f"n_components={n:<3} MSE={mean_squared_error(X, X_rec):.4f}  "
          f"variance kept={pca.explained_variance_ratio_.sum():.3f}")

rng = np.random.default_rng(42)
noisy = X + rng.normal(0, 1.0, X.shape)
pca = PCA(n_components=20, random_state=42)
denoised = pca.inverse_transform(pca.fit_transform(noisy))
print("denoise MSE (noisy vs clean):", round(mean_squared_error(X, noisy), 4))
print("denoise MSE (denoised vs clean):", round(mean_squared_error(X, denoised), 4))
