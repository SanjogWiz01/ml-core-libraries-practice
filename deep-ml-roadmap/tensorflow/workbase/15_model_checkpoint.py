import tensorflow as tf
print(tf.keras.callbacks.ModelCheckpoint('best_model.keras',monitor='val_loss',save_best_only=True))
