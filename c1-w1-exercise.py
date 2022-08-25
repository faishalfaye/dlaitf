import tensorflow as tf
import numpy as np
from tensorflow import keras

# GRADED FUNCTION: house_model
def house_model(y_new):
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)# Your Code Here#
    ys = np.array([50000.0, 100000.0, 150000.0, 200000.0, 250000.0, 300000.0], dtype=float)# Your Code Here#
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(input_shape=(1, ), units=1)
    ])
    model.compile(loss="mean_squared_error", optimizer="sgd")
    model.fit(xs, ys/100000, epochs=500)
    return model.predict(y_new)[0]

prediction = house_model([7.0])
print(prediction)

# %%javascript
# <!-- Save the notebook -->
# IPython.notebook.save_checkpoint();
#
# %%javascript
# IPython.notebook.session.delete();
# window.onbeforeunload = null
# setTimeout(function() { window.close(); }, 1000);