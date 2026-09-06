import tensorflow as tf
print(tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True))
