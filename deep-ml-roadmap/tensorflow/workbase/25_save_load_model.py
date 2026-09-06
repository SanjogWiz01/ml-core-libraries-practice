import tensorflow as tf
m=tf.keras.Sequential([tf.keras.layers.Input((5,)),tf.keras.layers.Dense(16,'relu'),tf.keras.layers.Dense(1)]); m.save('model.keras'); n=tf.keras.models.load_model('model.keras'); print(n)
