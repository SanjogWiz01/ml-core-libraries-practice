"""AUTOENCODERS — learn a compressed code by reconstructing the input.

An encoder maps x to latent code z; a decoder maps z back to x. Training
minimizes reconstruction loss, usually with a bottleneck or regularization to
discourage a trivial identity. The latent space is not automatically meaningful
or disentangled. Reconstruction error can rank anomalies, but expressive models
may reconstruct anomalies well too. This uses MLPRegressor with input-as-target
and a narrow hidden layer; deep versions are usually built in PyTorch/Keras.

IMPLEMENTATION: generate correlated features, scale them, train a network to
reproduce X, and compare row reconstruction errors. Its hidden layer is the
bottleneck representation.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(6)
t = rng.normal(size=(500, 2))
X = np.column_stack([t[:, 0], t[:, 1], t[:, 0]+t[:, 1],
                     2*t[:, 0]-t[:, 1], .5*t[:, 0]+rng.normal(0, .1, 500)])
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
autoencoder = MLPRegressor(hidden_layer_sizes=(2,), activation="tanh", solver="adam",
                           alpha=.01, max_iter=1500, random_state=6, early_stopping=True)
autoencoder.fit(Xs, Xs)  # reconstruction target equals input
reconstruction = autoencoder.predict(Xs)
error = np.mean((Xs-reconstruction)**2, axis=1)
print("Input shape / bottleneck size:", Xs.shape, autoencoder.hidden_layer_sizes)
print("Mean reconstruction MSE:", round(float(error.mean()), 4))
print("95th percentile MSE:", round(float(np.quantile(error, .95)), 4))
print("This compact example demonstrates a bottleneck, not a deep-learning benchmark.")
