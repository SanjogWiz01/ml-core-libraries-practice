import tensorflow as tf
X=tf.random.normal((500,6))*10; norm=tf.keras.layers.Normalization(); norm.adapt(X); m=tf.keras.Sequential([tf.keras.layers.Input((6,)),norm,tf.keras.layers.Dense(32,'relu'),tf.keras.layers.Dense(1)]); print(m(X[:2]))
