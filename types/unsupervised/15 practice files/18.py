"""18. Image colour quantization: compress an image with K-Means."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image

image = load_sample_image("china.jpg")
X = (image.reshape(-1, 3) / 255.0)

n_colors = 16
model = KMeans(n_clusters=n_colors, n_init=10, random_state=42).fit(X)

labels = model.predict(X)
compressed = model.cluster_centers_[labels].reshape(image.shape)
compressed_uint8 = (compressed * 255).astype(np.uint8)

print("original shape:  ", image.shape)
print("palette size:    ", n_colors, "colours")
print("distinct colours in compressed image:", len(np.unique(compressed_uint8.reshape(-1, 3), axis=0)))
print("bytes per pixel reduced from 24-bit to ~", int(np.ceil(np.log2(n_colors))), "-bit index")
