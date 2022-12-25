#https://stackoverflow.com/questions/71120275/data-cardinality-is-ambiguous
#https://stackoverflow.com/questions/63279168/valueerror-input-0-of-layer-sequential-is-incompatible-with-the-layer-expect

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

f = pd.read_csv(r'E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\NeuralNetworkingTests\Datasets\adj_cleaned_test_competition_ratios1-4.csv')

# Converting the each column to a list.
# Converting the list to floating value with for loop iteration.
relations = [float(i) for i in list(f['relation'])]
r1s = [float(i) for i in list(f['r1'])]
r2s = [float(i) for i in list(f['r2'])]
r3s = [float(i) for i in list(f['r3'])]
r4s = [float(i) for i in list(f['r4'])]

sets = [float(i) for i in list(f['set'])]
'''
x_data = np.array([[1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5],
                   [3, 5, 1, 3, 5, 1, 3, 5, 1, 3, 5, 1, 3, 5, 1, 3, 5, 1, 3, 5],
                   [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4]])
y_data = np.array([11, 6, 10, 6, 5, 19, 14, 14, 27, 15, 5, 9, 5, 5, 10, 15, 19, 21, 14, 12])
'''


x_data = np.array([relations, r1s, r2s, r3s, r4s])
print(x_data.shape)

x_data = x_data.reshape(x_data.shape[1],x_data.shape[0])
print(x_data.shape)
x_data = x_data.reshape(-1, 28, 28, 90300)


y_data = np.array(sets)
y_data = y_data.reshape(x_data.shape[0],1)
print(y_data.shape)

model = Sequential()
model.add(Dense(11, input_dim=3, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.45)
model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])

batch_size = 1
epochs = 500

result = model.fit(np.array(x_data), np.array(y_data), batch_size, epochs=epochs, shuffle=True, verbose=0)

result.history.keys()
plt.plot(result.history['loss'])