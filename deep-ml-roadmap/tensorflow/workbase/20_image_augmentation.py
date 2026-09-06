import tensorflow as tf
print(tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal'),tf.keras.layers.RandomRotation(.1),tf.keras.layers.RandomZoom(.1)]))
