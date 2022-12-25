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
    anchor_images.append(os.path.join(base_path,anchor_paths[i]))
    positive_images.append(os.path.join(base_path,positive_paths[i]))
    #negative_paths.append(os.path.join(base_path,negative_paths[i]))

image_count = len(anchor_images)

anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)

#negative_dataset = tf.data.Dataset.from_tensor_slices(negative_paths)

rng = np.random.RandomState(seed=42)
rng.shuffle(anchor_images)
rng.shuffle(positive_images)

negative_images = anchor_images+positive_images
np.random.RandomState(seed=32).shuffle(negative_images)

negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)
negative_dataset = negative_dataset.shuffle(buffer_size=4096)

dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.map(preprocess_triplets)

# Split the dataset into train and test
train_dataset = dataset.take(round(image_count * 0.8))
test_dataset = dataset.skip(round(image_count * 0.8))

train_dataset = train_dataset.batch(32, drop_remainder=False)
train_dataset = train_dataset.prefetch(8)

test_dataset = test_dataset.batch(32, drop_remainder=False)
test_dataset = test_dataset.prefetch(8)


def visualize(anchor, positive, negative):
    """Visualize a few triplets from the supplied batches."""

    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(9, 9))

    axs = fig.subplots(3, 3)
    for i in range(3):
        show(axs[i, 0], anchor[i])
        show(axs[i, 1], positive[i])
        show(axs[i, 2], negative[i])


visualize(*list(train_dataset.take(1).as_numpy_iterator())[0])