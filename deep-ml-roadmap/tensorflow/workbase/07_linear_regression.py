import tensorflow as tf, numpy as np
X=np.random.randn(500,1).astype('float32'); y=3*X+2+.2*np.random.randn(500,1).astype('float32')
m=tf.keras.Sequential([tf.keras.layers.Input((1,)),tf.keras.layers.Dense(1)]); m.compile('sgd','mse'); m.fit(X,y,epochs=30,verbose=0); print(m.predict([[1.]],verbose=0))
