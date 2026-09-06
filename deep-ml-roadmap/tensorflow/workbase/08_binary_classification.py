import tensorflow as tf, numpy as np
X=np.random.randn(1000,8).astype('float32'); y=(X[:,0]+X[:,1]>0).astype('float32')
m=tf.keras.Sequential([tf.keras.layers.Input((8,)),tf.keras.layers.Dense(32,activation='relu'),tf.keras.layers.Dropout(.2),tf.keras.layers.Dense(1,activation='sigmoid')]); m.compile('adam','binary_crossentropy',metrics=['accuracy']); m.fit(X,y,epochs=5,verbose=0); print(m.evaluate(X,y,verbose=0))
