# To ignore future warnings
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    # Packages
    import tensorflow as tf
    import numpy as np
    import tensorflow.keras as keras
    import pandas as pd
print('ready')

feature_names = ['index','val','primary_img','secondary_img','relation','r1','r2','r3','r4','set']
batch_size = 24
# Define Neural network
# Dense -> single neuron
# units = 1, so single
# input_shape[1] -> dimension of input data
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Compile the model with optimizer as SGD and loss function as Mean_Squared_Error
# Optimizer -> check how good the guess is
# Loss function -> the difference between the actual value and the predicted value, eventually loss value will reduces.
model.compile(optimizer='sgd',loss='mean_squared_error')
# Reading the csv file
f = pd.read_csv(r'E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\NeuralNetworkingTests\Datasets\adj_cleaned_test_competition_ratios1-4.csv')

# Converting the each column to a list.
# Converting the list to floating value with for loop iteration.
relations = [float(i) for i in list(f['relation'])]
r1s = [float(i) for i in list(f['r1'])]
r2s = [float(i) for i in list(f['r2'])]
r3s = [float(i) for i in list(f['r3'])]
r4s = [float(i) for i in list(f['r4'])]


train_X = {'Input1': relations, 'Input2': r1s, 'Input3': r2s, 'Input4': r3s, 'Input5': r4s}

sets = [float(i) for i in list(f['set'])]

print(r1s)
print(r2s)
print(sets)

#Xs = np.reshape([relations, r1s, r2s, r3s, r4s], (len(relations), 5))
Xs = [relations, r1s, r2s, r3s, r4s]
Xs = np.asarray(Xs)
Xs = Xs.reshape(Xs.shape[1], Xs.shape[0])

#Ys = np.asarray(sets)
print(Xs.shape)

model.fit(Xs,sets, epochs=50)

# Equation is y = x*5
# Print the prediction
print("Value for Y when X is 10.0: {}".format(model.predict([10.0])))


#create neural network using tensorflow to predict either 1 or 0 using 7 input values from a csv file
#create a neural network using tensorflow to predict either 1 or 0 using 7 input values from a csv file

