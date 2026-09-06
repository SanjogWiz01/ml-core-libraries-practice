import tensorflow as tf
base=tf.keras.applications.MobileNetV2(include_top=False,weights='imagenet',input_shape=(160,160,3)); base.trainable=False; inp=tf.keras.Input((160,160,3)); x=tf.keras.applications.mobilenet_v2.preprocess_input(inp); x=base(x,training=False); x=tf.keras.layers.GlobalAveragePooling2D()(x); out=tf.keras.layers.Dense(1,'sigmoid')(x); tf.keras.Model(inp,out).summary()
