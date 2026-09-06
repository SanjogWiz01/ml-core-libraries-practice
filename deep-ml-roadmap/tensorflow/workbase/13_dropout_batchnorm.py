import tensorflow as tf
m=tf.keras.Sequential([tf.keras.layers.Input((20,)),tf.keras.layers.Dense(64),tf.keras.layers.BatchNormalization(),tf.keras.layers.ReLU(),tf.keras.layers.Dropout(.3),tf.keras.layers.Dense(1)]); m.summary()
