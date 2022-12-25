'''
Create a Python script that generates a triplet-loss Siamese Neural Network
This model should take in three arrays of images: anchors, positives, negatives
The images each have a size of (108, 124) and three color channels
'''

# Import the necessary packages
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Lambda
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
import keras.backend as K
import numpy as np

batch_size = 1
epochs = 10

### TODO ###
# define/import anchors/positives/negatives arrays
# ASK what does the shape of the apn arrays need to be
# define batch size + epochs

# Define the triplet loss function
from random import random

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

path = "D:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\Data\scienceFairRawFacesNOBG"
csvPath = "D:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\SNN Test\Data\Triplet-Loss-Pairs.csv"

anchor_paths = []
positive_paths = []
negative_paths = []

anchor_images = []
positive_images = []
negative_images = []

def convertCSVtoImageArray(path, csvPath):
    global anchor_paths, positive_paths, negative_paths
    global anchor_images, positive_images, negative_images

    def getImagesFromCSV(csvPath):
        """
        Given the path to a csv file, return a list of the images in the csv file
        """
        csvFile = open(csvPath, "r")
        csvLines = csvFile.readlines()
        csvFile.close()
        for line in csvLines:
            a_l = (line.split(',')[0]).replace('\n', "")
            p_l = (line.split(',')[1]).replace('\n', "")
            n_l = (line.split(',')[2]).replace('\n', "")

            anchor_paths.append(a_l)
            positive_paths.append(p_l)
            negative_paths.append(n_l)

            #print(a_l, p_l, n_l)

        return anchor_paths, positive_paths, negative_paths

    getImagesFromCSV(csvPath)

    for image in range(1, len(anchor_paths)):  # ADD '_folder'
        #print(path + r"\noBG" + negative_paths[image])
        if image%100 == 0:
            print(image)
        a_i = cv2.imread(str(path + r"\noBG" + anchor_paths[image]))
        a_i = cv2.resize(a_i, (24, 32))

        p_i = cv2.imread(str(path + r"\noBG" + positive_paths[image]))
        p_i = cv2.resize(p_i, (24, 32))

        n_i = cv2.imread(str(path + r"\noBG" + negative_paths[image]))
        n_i = cv2.resize(n_i, (24, 32))

        anchor_images.append(a_i)
        positive_images.append(p_i)
        negative_images.append(n_i)

    anchor_images = np.array(anchor_images)
    anchor_images = anchor_images.astype('float32')

    positive_images = np.array(positive_images)
    positive_images = positive_images.astype('float32')

    negative_images = np.array(negative_images)
    negative_images = negative_images.astype('float32')

    print(anchor_images.shape)
    anchor_images /=255
    print(anchor_images.shape)

    print(positive_images.shape)
    positive_images /=255
    print(positive_images.shape)

    print(negative_images.shape)
    negative_images /=255
    print(negative_images.shape)

    return (anchor_images, positive_images, negative_images)

(anchors, positives, negatives) = convertCSVtoImageArray(path=path, csvPath=csvPath)


#x_train, x_test, y_train, y_test = train_test_split([anchors, positives, negatives], )

def triplet_loss(y_true, y_pred, alpha=0.2):
    '''
    y_true is not used in this function. It is only used to make the function
    compatible with the Keras API.
    '''

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=-1)

    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = K.sum(K.square(
        anchor - negative), axis=-1)

    # Step 3: Subtract the two previous distances and add alpha
    basic_loss = pos_dist - neg_dist + alpha

    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples
    loss = K.maximum(basic_loss, 0.0)

    return loss


# Define the shape of the input images
input_shape = (32, 24, 3)

# Define the anchor, positive, and negative inputs
#anchor_input = Input(input_shape, name='anchor_input')
#positive_input = Input(input_shape, name='positive_input')
#negative_input = Input(input_shape, name='negative_input')

anchor_input = anchors
positive_input = positives
negative_input = negatives

input_layer = Input(input_shape)
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

#model.add(Dense(4096, activation='sigmoid',
#                kernel_initializer='random_normal', bias_initializer='zeros'))

# Generate the encodings (feature vectors) for the three images
encoded_anchor = model(anchor_input)
encoded_positive = model(positive_input)
encoded_negative = model(negative_input)

# Create the three inputs
inputs = [anchor_input, positive_input, negative_input]

# Create the output layers
outputs = [encoded_anchor, encoded_positive, encoded_negative]

# Create the triplet network
triplet_model = Model(inputs, outputs, input_shape=(3, 10061, input_shape[0], input_shape[1], input_shape[2]))

# Compile the model
triplet_model.compile(loss=triplet_loss, optimizer=Adam(0.0001))

# Print a summary of the model
print(triplet_model.summary())

# Train the model
triplet_model.fit([anchor_input, positive_input, negative_input],
                  np.zeros(batch_size), batch_size=batch_size, epochs=epochs,
                  verbose=2)

# Explain what the y_pred parameter is in the triplet_loss function
'''
The y_pred parameter is a list of three arrays of images: anchors, positives,
and negatives.
'''
# What should the shape of the y_pred parameter be?
'''
The shape of the y_pred parameter should be (3, 108, 124, 3).
'''
# What should the shape of the y_true parameter be?
'''
The shape of the y_true parameter should be (3, 108, 124, 3).
'''
# What is the purpose of the alpha parameter?
'''
The alpha parameter is used to add a margin to the loss function.
'''
# What is the purpose of the pos_dist and neg_dist variables?
'''
The pos_dist and neg_dist variables are used to compute the distance between
the anchor and the positive and the anchor and the negative.
'''
# What is the purpose of the basic_loss variable?
'''
The basic_loss variable is used to subtract the two previous distances and add
alpha.
'''
# What is the purpose of the loss variable?
'''
The loss variable is used to take the maximum of basic_loss and 0.0.
'''
# What is the purpose of the encoded_anchor, encoded_positive, and encoded_negative
# variables?
'''
The encoded_anchor, encoded_positive, and encoded_negative variables are used to
generate the encodings (feature vectors) for the three images.
'''
# What is the purpose of the inputs variable?
'''
The inputs variable is used to create the three inputs.
'''
# What is the purpose of the outputs variable?
'''
The outputs variable is used to create the output layers.
'''
# What is the purpose of the triplet_model variable?
'''
The triplet_model variable is used to create the triplet network.
'''
