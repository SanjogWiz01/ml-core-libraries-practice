import tensorflow as tf
print('GPUs:',tf.config.list_physical_devices('GPU'))
if tf.config.list_physical_devices('GPU'): tf.keras.mixed_precision.set_global_policy('mixed_float16'); print(tf.keras.mixed_precision.global_policy())
