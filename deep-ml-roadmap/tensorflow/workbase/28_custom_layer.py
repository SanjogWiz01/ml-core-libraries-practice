import tensorflow as tf
class SquareLayer(tf.keras.layers.Layer):
 def call(self,x): return tf.square(x)
print(SquareLayer()(tf.constant([2.,3.])))
