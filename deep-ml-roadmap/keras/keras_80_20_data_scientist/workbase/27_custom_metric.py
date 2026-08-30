import keras

@keras.utils.register_keras_serializable()
class MeanAbsolutePercentageError(keras.metrics.Metric):
    def __init__(self, name="custom_mape", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = keras.ops.abs((y_true - y_pred) / keras.ops.maximum(keras.ops.abs(y_true), 1e-7))
        self.total.assign_add(keras.ops.sum(error))
        self.count.assign_add(keras.ops.cast(keras.ops.size(error), "float32"))

    def result(self):
        return 100 * self.total / self.count

    def reset_state(self):
        self.total.assign(0)
        self.count.assign(0)

print("Custom metric defined.")
