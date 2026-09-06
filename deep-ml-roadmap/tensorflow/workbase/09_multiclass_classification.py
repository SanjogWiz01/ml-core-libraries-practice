import tensorflow as tf, numpy as np
X=np.random.randn(1200,10).astype('float32'); y=np.random.randint(0,4,1200)
m=tf.keras.Sequential([tf.keras.layers.Input((10,)),tf.keras.layers.Dense(64,activation='relu'),tf.keras.layers.Dense(4,activation='softmax')]); m.compile('adam','sparse_categorical_crossentropy',metrics=['accuracy']); m.fit(X,y,epochs=5,verbose=0); print(m.evaluate(X,y,verbose=0))
