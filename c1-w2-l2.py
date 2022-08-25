import tensorflow as tf
from tensorflow import keras

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.99):
            print("dah cukup sampe sini")
            self.model.stop_training = True

callbacks = myCallback()


mnist = tf.keras.datasets.mnist
(x_train_full, y_train_full), (x_test_full, y_test_full) = mnist.load_data()
# print(x_train_full.shape)
# print(y_train_full.shape)
# print(x_train_full[0])
# print(y_train_full[:])

x_validation, x_train = x_train_full[:5000]/255.0, x_train_full[5000:]/255.0
y_validation, y_train = y_train_full[:5000], y_train_full[5000:]

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_validation, y_validation), callbacks=[callbacks])

import matplotlib.pyplot as plt

acc = history.history["accuracy"]
loss = history.history["loss"]
val_acc = history.history["val_accuracy"]
val_loss = history.history["val_loss"]
epochs = range(len(history.history["accuracy"]))


plt.plot(epochs, acc, label="accuracy")
plt.plot(epochs, loss, label="loss")
plt.plot(epochs, val_acc, label="val_acc")
plt.plot(epochs, val_loss, label="val_loss")
plt.xlabel("EPOCHS")
plt.ylabel("ACCURACY")
plt.title("KAGA ADE JUDULNYE")
plt.legend()
plt.grid(b=True)
plt.show()
