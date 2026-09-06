import tensorflow as tf
X=tf.random.normal((100,5)); y=tf.random.uniform((100,),maxval=2,dtype=tf.int32); ds=tf.data.Dataset.from_tensor_slices((X,y)).shuffle(100).batch(16).prefetch(tf.data.AUTOTUNE)
for x,y in ds.take(1): print(x.shape,y.shape)
