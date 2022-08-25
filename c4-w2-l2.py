import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)



def plot_series(time, series, format="-", start=0, end=None):
  plt.plot(time[start:end], series[start:end], format)
  plt.xlabel("time")
  plt.ylabel("value")
  plt.grid(True)



def trend(time, slope=0):
  return slope*time



def seasonal_pattern(season_time):
  return np.where(season_time < 0.4,
                  np.cos(season_time * 2 * np.pi),
                  1 / np.exp(3*season_time))



def seasonality(time, period, amplitude=1, phase=0):
  season_time = ((time + phase) % period) / period
  return amplitude * seasonal_pattern(season_time)



def noise(time, noise_level=1, seed=None):
  rnd = np.random.RandomState(seed)
  return rnd.randn(len(time)) * noise_level



time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

print(time)
print(np.arange(4*365+1, dtype="float32"))



#create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
#update with noise
series += noise(time, noise_level, seed=42)

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window : window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset


dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
print(dataset)
l0 = tf.keras.layers.Dense(1, input_shape=[window_size])
model = tf.keras.models.Sequential([l0])


model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
model.fit(dataset, epochs=100, verbose=0)


#the line below is printing w1, w2, ..., w19, b
#b is bias
print("Layer weights {}".format(l0.get_weights()))


#len(series) = 1461 = len(time)

#series is numpy array
#series[np.newaxis] is for adding 1 dimension to the array
#before after np.newaxis : [1, 2, 3] -> [[1, 2, 3]]
print(series[np.newaxis])
print(len(series))
plot_series(time_train, x_train)


forecast = []

for time in range(len(series) - window_size):
  forecast.append(model.predict(series[time : time + window_size][np.newaxis]))
  #len forecast = 1441
  #series[0:20][np.newaxis] -> series[1:21] -> series[2:22]
  #sampe series[1441:1461] PREDICTNYA
  #INPUT : x1, x2, ..., x20
  #OUTPUT : y = f(x1, x2, ..., x20)

# forecast[980:]
forecast = forecast[split_time - window_size:] #len(forecast) = 461
results = np.array(forecast)[:, 0, 0] #len(results) = 461

# forecast : [array([[65.804146]], dtype=float32), .. , array([[89.09347]], dtype=float32)]
# forecast was a python list, then transformed to np.array
# np.array(forecast) : [ [[79.43552]], [[ 82.013725]], .., [[ 89.09347 ]] ]
# results : [65.804146  67.3648 .. 89.09347]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, results)


tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()