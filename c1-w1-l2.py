import tensorflow as tf
import numpy as np
from tensorflow import keras


x = np.array([-1.0, 0, 1.0, 2.0, 3.0, 4.0])
y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])
model.compile(optimizer="sgd", loss="mean_squared_error")
model.fit(x,y, epochs=501)
print(model.predict([10.0]))