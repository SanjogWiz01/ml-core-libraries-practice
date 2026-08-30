import keras
import numpy as np

model = keras.Sequential([
    keras.Input(shape=(3,)),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

X = np.random.randn(100,3).astype("float32")
y = np.random.randint(0,2,100)
model.fit(X,y,epochs=2,verbose=0)

print("Evaluation:", model.evaluate(X,y,verbose=0))
print("Predictions:", model.predict(X[:5], verbose=0))
