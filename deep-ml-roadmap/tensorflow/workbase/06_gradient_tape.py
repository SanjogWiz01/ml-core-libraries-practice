import tensorflow as tf
x=tf.Variable(3.)
with tf.GradientTape() as tape: y=x**2+2*x
print(y.numpy(),tape.gradient(y,x).numpy())
