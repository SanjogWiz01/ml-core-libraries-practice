import tensorflow as tf
w=tf.Variable(1.); w.assign_add(2.); print(w.numpy())
