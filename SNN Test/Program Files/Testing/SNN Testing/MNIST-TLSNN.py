#https://towardsdatascience.com/how-to-train-your-siamese-neural-network-4c6da3259463

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import glob

from datetime import datetime
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import Callback, CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, Input, concatenate
from tensorflow.keras.layers import Layer, BatchNormalization, MaxPooling2D, Concatenate, Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform, he_uniform
from tensorflow.keras.regularizers import l2
#from tensorflow.keras.utils import multi_gpu_model


from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score
import math
#from pylab import dist
import json

from tensorflow.python.client import device_lib
import matplotlib.gridspec as gridspec

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

num_classes = len(np.unique(y_train))

x_train_w = x_train.shape[1] # (60000, 28, 28)
x_train_h = x_train.shape[2]
x_test_w = x_test.shape[1]
x_test_h = x_test.shape[2]

x_train_w_h = x_train_w * x_train_h # 28 * 28 = 784
x_test_w_h = x_test_w * x_test_h

x_train = np.reshape(x_train, (x_train.shape[0], x_train_w_h))/255. # (60000, 784)
x_test = np.reshape(x_test, (x_test.shape[0], x_test_w_h))/255.


def create_batch(batch_size=256, split="train"):
    x_anchors = np.zeros((batch_size, x_train_w_h))
    x_positives = np.zeros((batch_size, x_train_w_h))
    x_negatives = np.zeros((batch_size, x_train_w_h))

    if split == "train":
        data = x_train
        data_y = y_train
    else:
        data = x_test
        data_y = y_test

    for i in range(0, batch_size):
        # We need to find an anchor, a positive example and a negative example
        random_index = random.randint(0, data.shape[0] - 1)
        x_anchor = data[random_index]
        y = data_y[random_index]

        indices_for_pos = np.squeeze(np.where(data_y == y))
        indices_for_neg = np.squeeze(np.where(data_y != y))

        x_positive = data[indices_for_pos[random.randint(0, len(indices_for_pos) - 1)]]
        x_negative = data[indices_for_neg[random.randint(0, len(indices_for_neg) - 1)]]

        x_anchors[i] = x_anchor
        x_positives[i] = x_positive
        x_negatives[i] = x_negative

    return [x_anchors, x_positives, x_negatives]


def create_hard_batch(batch_size, num_hard, split="train"):
    x_anchors = np.zeros((batch_size, x_train_w_h))
    x_positives = np.zeros((batch_size, x_train_w_h))
    x_negatives = np.zeros((batch_size, x_train_w_h))

    if split == "train":
        data = x_train
        data_y = y_train
    else:
        data = x_test
        data_y = y_test

    # Generate num_hard number of hard examples:
    hard_batches = []
    batch_losses = []

    rand_batches = []

    # Get some random batches
    for i in range(0, batch_size):
        hard_batches.append(create_batch(1, split))

        A_emb = embedding_model.predict(hard_batches[i][0])
        P_emb = embedding_model.predict(hard_batches[i][1])
        N_emb = embedding_model.predict(hard_batches[i][2])

        # Compute d(A, P) - d(A, N) for each selected batch
        batch_losses.append(np.sum(np.square(A_emb - P_emb), axis=1) - np.sum(np.square(A_emb - N_emb), axis=1))

    # Sort batch_loss by distance, highest first, and keep num_hard of them
    hard_batch_selections = [x for _, x in sorted(zip(batch_losses, hard_batches), key=lambda x: x[0])]
    hard_batches = hard_batch_selections[:num_hard]

    # Get batch_size - num_hard number of random examples
    num_rand = batch_size - num_hard
    for i in range(0, num_rand):
        rand_batch = create_batch(1, split)
        rand_batches.append(rand_batch)

    selections = hard_batches + rand_batches

    for i in range(0, len(selections)):
        x_anchors[i] = selections[i][0]
        x_positives[i] = selections[i][1]
        x_negatives[i] = selections[i][2]

    return [x_anchors, x_positives, x_negatives]


def create_embedding_model(emb_size):
    embedding_model = tf.keras.models.Sequential([
        Dense(4096,
              activation='relu',
              kernel_regularizer=l2(1e-3),
              kernel_initializer='he_uniform',
              input_shape=(x_train_w_h,)),
        Dense(emb_size,
              activation=None,
              kernel_regularizer=l2(1e-3),
              kernel_initializer='he_uniform')
    ])

    embedding_model.summary()

    return embedding_model


def create_embedding_model(emb_size):
    embedding_model = tf.keras.models.Sequential([
        Dense(4096,
              activation='relu',
              kernel_regularizer=l2(1e-3),
              kernel_initializer='he_uniform',
              input_shape=(x_train_w_h,)),
        Dense(emb_size,
              activation=None,
              kernel_regularizer=l2(1e-3),
              kernel_initializer='he_uniform')
    ])

    embedding_model.summary()

    return embedding_model

def create_embedding_model(emb_size):
    embedding_model = tf.keras.models.Sequential([
        Dense(4096,
              activation='relu',
              kernel_regularizer=l2(1e-3),
              kernel_initializer='he_uniform',
              input_shape=(x_train_w_h,)),
        Dense(emb_size,
              activation=None,
              kernel_regularizer=l2(1e-3),
              kernel_initializer='he_uniform')
    ])

    embedding_model.summary()

    return embedding_model

def triplet_loss(y_true, y_pred):
    anchor, positive, negative = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size],y_pred[:,2*emb_size:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return tf.maximum(positive_dist - negative_dist + alpha, 0.)


def data_generator(batch_size=256, num_hard=50, split="train"):
    while True:
        x = create_hard_batch(batch_size, num_hard, split)
        y = np.zeros((batch_size, 3*emb_size))
        yield x, y


# Hyperparams
batch_size = 256
epochs = 100
steps_per_epoch = int(x_train.shape[0] / batch_size)
val_steps = int(x_test.shape[0] / batch_size)
alpha = 0.2
num_hard = int(batch_size * 0.5)  # Number of semi-hard triplet examples in the batch
lr = 0.00006
optimiser = 'Adam'
emb_size = 10

with tf.device("/cpu:0"):
    # Create the embedding model
    print("Generating embedding model... \n")
    embedding_model = create_embedding_model(emb_size)

    print("\nGenerating SNN... \n")
    # Create the SNN
    siamese_net = create_SNN(embedding_model)
    # Compile the SNN
    optimiser_obj = Adam(lr=lr)
    siamese_net.compile(loss=triplet_loss, optimizer=optimiser_obj)

    # Store visualisations of the embeddings using PCA for display
    # Create representations of the embedding space via PCA
    embeddings_before_train = loaded_emb_model.predict(x_test[:500, :])
    pca = PCA(n_components=2)
    decomposed_embeddings_before = pca.fit_transform(embeddings_before_train)