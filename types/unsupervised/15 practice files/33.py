"""33. FastICA: blind source separation of mixed independent signals."""
import numpy as np
from sklearn.decomposition import FastICA

rng = np.random.default_rng(42)
t = np.linspace(0, 8, 2000)
sources = np.c_[np.sin(2 * t), np.sign(np.sin(3 * t)), rng.uniform(-1, 1, 2000)]

mixing = np.array([[1.0, 1.0, 1.0],
                   [0.5, 2.0, 1.0],
                   [1.5, 1.0, 2.0]])
observed = sources @ mixing.T

ica = FastICA(n_components=3, random_state=42, max_iter=1000)
recovered = ica.fit_transform(observed)

def corr(a, b):
    return abs(np.corrcoef(a, b)[0, 1])

print("max |correlation| between each recovered signal and its true source:")
for i in range(3):
    best = max(corr(recovered[:, i], sources[:, j]) for j in range(3))
    print(f"  component {i}: {best:.3f}")
