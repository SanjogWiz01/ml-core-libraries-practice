import tensorflow as tf
x=tf.random.normal((100,4)); y=tf.random.normal((100,1)); m=tf.keras.Sequential([tf.keras.layers.Input((4,)),tf.keras.layers.Dense(16,'relu'),tf.keras.layers.Dense(1)]); loss_fn=tf.keras.losses.MeanSquaredError(); opt=tf.keras.optimizers.Adam()
for _ in range(5):
 with tf.GradientTape() as tape: loss=loss_fn(y,m(x,training=True))
 opt.apply_gradients(zip(tape.gradient(loss,m.trainable_variables),m.trainable_variables))
print(loss.numpy())
