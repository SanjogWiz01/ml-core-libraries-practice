import keras

class SquareLayer(keras.layers.Layer):
    def call(self, inputs):
        return inputs ** 2

layer = SquareLayer()
print(layer(3.0))
