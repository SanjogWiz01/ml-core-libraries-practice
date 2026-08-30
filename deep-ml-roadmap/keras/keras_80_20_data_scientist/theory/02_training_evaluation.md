# Training, Evaluation & Experimentation

## 80/20 concepts

### Compile
`compile()` connects the architecture to the learning configuration.

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)
```

### Fit
```python
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32
)
```

Important arguments:
- `epochs`
- `batch_size`
- `validation_split`
- `validation_data`
- `callbacks`
- `verbose`

### Evaluate and predict
```python
model.evaluate(X_test, y_test)
probabilities = model.predict(X_test)
```

### Loss selection
- Regression: MSE, MAE, Huber
- Binary classification: binary cross-entropy
- Multi-class integer labels: sparse categorical cross-entropy
- Multi-class one-hot labels: categorical cross-entropy

### Metrics
Classification:
- Accuracy
- Precision
- Recall
- AUC
- F1 (often calculated outside Keras when appropriate)

Regression:
- MAE
- MSE
- RMSE
- R² (commonly computed with sklearn)

### Overfitting
Typical signals:
- training loss keeps falling
- validation loss starts rising
- training accuracy increases while validation performance stalls/falls

Solutions:
- more data
- simpler architecture
- dropout
- regularization
- early stopping
- data augmentation for images
- transfer learning

### Callbacks
The highest-value callbacks are:
- `EarlyStopping`
- `ModelCheckpoint`
- `ReduceLROnPlateau`
- `TensorBoard`

### Reproducible experimentation
Keep:
- random seeds
- preprocessing rules
- architecture
- hyperparameters
- metrics
- training history
- model artifact
- experiment notes

A good data scientist treats every training run as an experiment.
