import keras
import numpy as np

# Synthetic tabular binary-classification workflow.
rng = np.random.default_rng(42)
X = rng.normal(size=(2000, 12)).astype("float32")
y = (X[:, 0] + 0.7*X[:, 1] - 0.4*X[:, 2] > 0).astype("float32")

normalizer = keras.layers.Normalization()
normalizer.adapt(X[:1600])

model = keras.Sequential([
    keras.Input(shape=(12,)),
    normalizer,
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
]

history = model.fit(
    X[:1600], y[:1600],
    validation_data=(X[1600:], y[1600:]),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=0
)

print(model.evaluate(X[1600:], y[1600:], verbose=0))
model.save("final_model.keras")
