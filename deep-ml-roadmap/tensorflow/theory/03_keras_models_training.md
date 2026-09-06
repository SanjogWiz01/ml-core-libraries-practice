# Keras Models & Training

## Standard workflow
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=.2, epochs=20)
model.evaluate(X_test, y_test)
model.predict(X_test)
```

Regression: MSE/MAE/Huber. Binary: binary cross-entropy. Multiclass: sparse/categorical cross-entropy.

Important callbacks: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, TensorBoard.

For images use CNNs and augmentation. For sequences use LSTM/GRU. For text use embeddings and sequence models.