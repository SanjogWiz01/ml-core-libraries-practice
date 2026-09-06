import tensorflow as tf
m=tf.keras.Sequential([tf.keras.layers.Input((64,64,3)),tf.keras.layers.Rescaling(1/255.),tf.keras.layers.Conv2D(32,3,'same',activation='relu'),tf.keras.layers.MaxPooling2D(),tf.keras.layers.Conv2D(64,3,'same',activation='relu'),tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(10,'softmax')]); print(m(tf.random.uniform((8,64,64,3))).shape)
