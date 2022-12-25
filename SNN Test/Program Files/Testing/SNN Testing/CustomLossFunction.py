import tensorflow as tf
import numpy as np

def my_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)])

model.compile(optimizer='adam', loss=my_loss_fn)

x = np.random.rand(1000)
y = x**2

history = model.fit(x, y, epochs=10)