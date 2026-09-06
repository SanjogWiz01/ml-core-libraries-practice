import tensorflow as tf, numpy as np
X=np.random.randn(300,4).astype('float32'); y=(X[:,0]>0).astype('float32'); m=tf.keras.Sequential([tf.keras.layers.Input((4,)),tf.keras.layers.Dense(16,'relu'),tf.keras.layers.Dense(1,'sigmoid')]); m.compile('adam','binary_crossentropy',metrics=['accuracy']); h=m.fit(X,y,epochs=3,verbose=0); print(h.history.keys())
