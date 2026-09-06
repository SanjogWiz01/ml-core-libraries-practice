import tensorflow as tf
X=tf.random.normal((100,4)); y=tf.zeros((100,)); ds=tf.data.Dataset.from_tensor_slices((X,y)).map(lambda x,y:(x*2,y),num_parallel_calls=tf.data.AUTOTUNE).batch(16); print(next(iter(ds))[0].shape)
