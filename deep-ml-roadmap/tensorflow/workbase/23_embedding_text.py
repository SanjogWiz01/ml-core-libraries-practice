import tensorflow as tf
m=tf.keras.Sequential([tf.keras.layers.Input((100,),dtype='int32'),tf.keras.layers.Embedding(10000,64),tf.keras.layers.GlobalAveragePooling1D(),tf.keras.layers.Dense(1,'sigmoid')]); print(m(tf.random.uniform((8,100),maxval=10000,dtype=tf.int32)).shape)
