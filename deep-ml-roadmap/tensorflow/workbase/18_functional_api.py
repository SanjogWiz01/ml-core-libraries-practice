import tensorflow as tf
inp=tf.keras.Input((10,)); x=tf.keras.layers.Dense(64,'relu')(inp); x=tf.keras.layers.Dropout(.2)(x); out=tf.keras.layers.Dense(1)(x); tf.keras.Model(inp,out).summary()
