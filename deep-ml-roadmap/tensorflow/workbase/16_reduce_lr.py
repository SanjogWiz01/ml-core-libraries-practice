import tensorflow as tf
print(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=3,min_lr=1e-6))
