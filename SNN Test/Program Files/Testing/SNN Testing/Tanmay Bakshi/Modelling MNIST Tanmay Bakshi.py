import random

import keras.metrics
import matplotlib.pyplot as plt
import numpy as np
import umap
import cv2
from keras import Input
#from tensorflow.keras import backend as K
#from tensorflow.python.keras.datasets import mnist
#from tensorflow.keras.models import Model
#from tensorflow.python.keras.layers import *
#from tensorflow.python.keras.layers import Conv2D

from keras import backend as K
from keras.datasets import mnist
from keras.models import Model
from keras.layers import *

(x_train, y_train), (x_test, y_test) = mnist.load_data()

dataset = mnist.load_data()
print(type(dataset))
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))
print(type(dataset[1]))
print(type(dataset))
print("--------------------------")
#print(dataset[0][0])
#input("")

#dataset[0] is tuple of (x_train (images), y_train (classes)) and dataset[1] is tuple of (x_test (images), y_test (classes))
#dataset[2] is out of range because there are two tuples: train and test

#dataset = ((x_train, y_train), (x_test, y_test))
#x_train = nd.array
#hierarchy = tuple>2tuples>2ndarrays in each tuple

print(x_train.shape)
print(y_train.shape)
for i in y_train:
    print(i)
print(x_test.shape)
print(y_test.shape)
print(type(x_train))

x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.

classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]

def data_generator(batch_size=64):
    while True:
        a = []
        p = []
        n = []
        for _ in range(batch_size):
            pos_neg = random.sample(classes, 2)
            positive_samples = random.sample(list(x_train[y_train == pos_neg[0]]), 2)
            negative_sample = random.choice(list(x_train[y_train == pos_neg[1]]))
            a.append(positive_samples[0])
            p.append(positive_samples[1])
            n.append(negative_sample)
        yield ([np.array(a), np.array(p), np.array(n)], np.zeros((batch_size, 1)).astype("float32"))

def triplet_loss(y_true, y_pred):
    anchor_out = y_pred[:, 0:100]
    positive_out = y_pred[:, 100:200]
    negative_out = y_pred[:, 200:300]

    pos_dist = K.sum(K.abs(anchor_out - positive_out), axis=1) #l1 dist between anchor <-> positive
    neg_dist = K.sum(K.abs(anchor_out - negative_out), axis=1)  # l1 dist between anchor <-> positive

    probs = K.softmax([pos_dist, neg_dist], axis = 0)

    return K.mean(K.abs(probs[0]) + K.abs(1.0 - probs[1]))

input_layer = Input((28, 28, 1))
x = Conv2D(32, 3, activation="relu")(input_layer)
x = Conv2D(32, 3, activation="relu")(x)
x = MaxPool2D(2)(x)
x = Conv2D(64, 3, activation="relu")(input_layer)
x = Conv2D(64, 3, activation="relu")(x)
x = MaxPool2D(2)(x)
x = Conv2D(128, 3, activation="relu")(x)
x = Flatten()(x)
x = Dense(100, activation="relu")(x)
model = Model(input_layer, x)
model.summary()

triplet_model_a = Input((28, 28, 1))
triplet_model_p = Input((28, 28, 1))
triplet_model_n = Input((28, 28, 1))

triplet_model_out = Concatenate()([model(triplet_model_a), model(triplet_model_p), model(triplet_model_n)])
triplet_model = Model([triplet_model_a, triplet_model_p, triplet_model_n], triplet_model_out)
triplet_model.summary()

triplet_model.compile(loss=triplet_loss, optimizer="adam", metrics=['accuracy', keras.metrics.BinaryAccuracy()])
cv2.imshow("0", x_train[0])
cv2.imshow("1", x_train[1])

cv2.waitKey()
triplet_model.fit(data_generator(), steps_per_epoch=150, epochs=3)
triplet_model.save("MNISTtriplet.h5")
model_embeddings = triplet_model.layers[3].predict(x_test, verbose=1)
print(model_embeddings.shape)


reduced_embeddings = umap.UMAP(n_neighbors = 15, min_dist = 0.3, metric = 'correlaton').fit_transform(model_embeddings)
print(reduced_embeddings.shape)

plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:,1], c=y_test)