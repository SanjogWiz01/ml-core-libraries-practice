import tensorflow as tf
x=tf.random.normal((2,3)); y=tf.random.normal((2,3)); print(x+y); print(x*y); print(tf.matmul(x,y,transpose_b=True)); print(tf.reshape(x,(3,2)))
