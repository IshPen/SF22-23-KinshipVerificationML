#https://keras.io/examples/vision/siamese_network/
#code from above link

import matplotlib.pyplot as plt
import numpy as np
import os
import random

import tensorflow as tf
from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
import cv2

import pandas as pd

base_path = "E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\Data\scienceFairRawFacesNOBG"
target_shape = (108, 124)

#read csv columns from E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\SNN Test\Data\Triplet-Loss-Pairs.csv

pairs = pd.read_csv("E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\SNN Test\Data\Triplet-Loss-Pairs.csv")
print(pairs.values)

anchor_paths = []
positive_paths = []
negative_paths = []

for i in range(len(pairs)):
    anchor_paths.append("noBG" + str(pairs.values[i][0]))
    positive_paths.append("noBG" + str(pairs.values[i][1]))
    negative_paths.append("noBG" + str(pairs.values[i][2]))

def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image

def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """

    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )

anchor_images = []
positive_images = []
negative_images = []


for i in range(len(anchor_paths)):
    anchor_images.append(cv2.imread(os.path.join(base_path,anchor_paths[i])))
    positive_images.append(cv2.imread(os.path.join(base_path,positive_paths[i])))
    negative_images.append(cv2.imread(os.path.join(base_path,negative_paths[i])))

image_count = len(anchor_images)

#cv2.imshow("anchor", anchor_paths[2])
#cv2.imshow("positive", positive_paths[2])
#cv2.imshow("negative", negative_paths[2])
#cv2.waitKey(0)
base_cnn = resnet.ResNet50(
    weights="imagenet", input_shape=target_shape + (3,), include_top=False
)

flatten = layers.Flatten()(base_cnn.output)
dense1 = layers.Dense(512, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(256, activation="relu")(dense1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(256)(dense2)

embedding = Model(base_cnn.input, output, name="Embedding")

trainable = False
for layer in base_cnn.layers:
    if layer.name == "conv5_block1_out":
        trainable = True
    layer.trainable = trainable


class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
positive_input = layers.Input(name="positive", shape=target_shape + (3,))
negative_input = layers.Input(name="negative", shape=target_shape + (3,))

distances = DistanceLayer()(
    embedding(resnet.preprocess_input(anchor_input)),
    embedding(resnet.preprocess_input(positive_input)),
    embedding(resnet.preprocess_input(negative_input)),
)

siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)

class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]
#create a train test split
train_anchor_images = []
train_positive_images = []
train_negative_images = []

test_anchor_images = []
test_positive_images = []
test_negative_images = []

for i in range(image_count):
    if i%10 == 0:
        test_anchor_images.append(anchor_images[i])
        test_positive_images.append(positive_images[i])
        test_negative_images.append(negative_images[i])
    else:
        train_anchor_images.append(anchor_images[i])
        train_positive_images.append(positive_images[i])
        train_negative_images.append(negative_images[i])


train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_anchor_images, train_positive_images, train_negative_images)
)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_anchor_images, test_positive_images, test_negative_images)
)
test_dataset = test_dataset.batch(32)
#train the model

siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(0.0001))
siamese_model.fit(train_dataset, epochs=10, validation_data=test_dataset)
'''
rng = np.random.RandomState(seed=42)
rng.shuffle(anchor_paths)
rng.shuffle(positive_paths)

negative_paths = anchor_paths+positive_paths
np.random.RandomState(seed=32).shuffle(negative_paths)


negative_dataset = tf.data.Dataset.from_tensor_slices(negative_paths)
negative_dataset = negative_dataset.shuffle(buffer_size=4096)
'''
