import tensorflow as tf
m=tf.keras.Sequential([tf.keras.layers.Input((5,)),tf.keras.layers.Dense(16),tf.keras.layers.Dense(1)]); m.save_weights('weights.weights.h5'); n=tf.keras.Sequential([tf.keras.layers.Input((5,)),tf.keras.layers.Dense(16),tf.keras.layers.Dense(1)]); n.load_weights('weights.weights.h5'); print('restored')
