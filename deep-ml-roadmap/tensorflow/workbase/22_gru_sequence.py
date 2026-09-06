import tensorflow as tf
m=tf.keras.Sequential([tf.keras.layers.Input((30,8)),tf.keras.layers.GRU(64),tf.keras.layers.Dense(1)]); print(m(tf.random.normal((16,30,8))).shape)
