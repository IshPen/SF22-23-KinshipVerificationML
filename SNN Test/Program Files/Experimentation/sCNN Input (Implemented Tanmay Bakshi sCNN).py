import random

import keras
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(7)
import umap
from keras import Input
#from tensorflow.keras import backend as K
#from tensorflow.python.keras.datasets import mnist
#from tensorflow.keras.models import Model
#from tensorflow.python.keras.layers import *
#from tensorflow.python.keras.layers import Conv2D

from keras import backend as K
#from keras.datasets import mnist
from keras.models import Model
from keras.layers import *
import cv2
import time
from contrast import adjust_contrast, increase_brightness

from sklearn.model_selection import train_test_split
from facialTriangulationAvg import triangulation
from sklearn import metrics
resize = (28, 28)

path = "D:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\Data\scienceFairRawFacesNOBG"
simplified_path = "D:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\Data\scienceFairTriangulatedFaces"
csvPath = "D:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\SNN Test\Data\Triplet-Loss-Pairs.csv"

anchor_paths = []
positive_paths = []
negative_paths = []

anchor_images = []
positive_images = []
negative_images = []

def convertCSVtoImageArray(path, csvPath, simplified, contrast, brightness):
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
        return anchor_paths, positive_paths, negative_paths

    getImagesFromCSV(csvPath)

    for image in range(0, len(anchor_paths)):
        if image%100 == 0:
            print(image)

        ## IMAGE SOURCING ##

        if simplified:
            a_i = cv2.imread(str(simplified_path + r"\simplifiedF" + (anchor_paths[image])[1:]))
            a_i = cv2.resize(a_i, resize)

            p_i = cv2.imread(str(simplified_path + r"\simplifiedF" + (positive_paths[image])[1:]))
            p_i = cv2.resize(p_i, resize)

            n_i = cv2.imread(str(simplified_path + r"\simplifiedF" + (negative_paths[image])[1:]))
            n_i = cv2.resize(n_i, resize)

        elif not simplified:
            a_i = cv2.imread(str(path + r"\noBG" + anchor_paths[image]))
            a_i = cv2.resize(a_i, resize)
            # print(str(simplified_path + r"\simplifiedF" + (anchor_paths[image])[1:]))

            p_i = cv2.imread(str(path + r"\noBG" + positive_paths[image]))
            p_i = cv2.resize(p_i, resize)

            n_i = cv2.imread(str(path + r"\noBG" + negative_paths[image]))
            n_i = cv2.resize(n_i, resize)
        if contrast > 1:
            a_i = adjust_contrast(a_i, contrast_factor=contrast)
            p_i = adjust_contrast(p_i, contrast_factor=contrast)
            n_i = adjust_contrast(n_i, contrast_factor=contrast)

        #a_iprev = a_i
        a_i = increase_brightness(a_i, value=brightness)
        p_i = increase_brightness(p_i, value=brightness)
        n_i = increase_brightness(n_i, value=brightness)
        #a_i = cv2.resize(a_i, (a_i.shape[0]*15, a_i.shape[1]*15))
        #a_iprev = cv2.resize(a_iprev, (a_iprev.shape[0]*15, a_iprev.shape[1]*15))
        #cv2.imshow("a_i", a_i)
        #cv2.imshow("aiprev", a_iprev)
        #cv2.waitKey()
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


# Below data_generator is not used
def data_generator(batch_size=64):
    while True:
        a = []
        p = []
        n = []
        for _ in range(batch_size):
            arr_loc = random.choice(y_train)
            positive_samples = [anchors[arr_loc], positives[arr_loc]]
            negative_sample = negatives[arr_loc]
            a.append(positive_samples[0])
            p.append(positive_samples[1])
            n.append(negative_sample)
        yield ([np.array(a), np.array(p), np.array(n)], np.zeros((batch_size, 1)).astype("float32"))

#triplet_loss function
def triplet_loss(y_true, y_pred):
    anchor_out = y_pred[:, 0:100]
    positive_out = y_pred[:, 100:200]
    negative_out = y_pred[:, 200:300]

    pos_dist = K.sum(K.abs(anchor_out - positive_out), axis=1)  # l1 dist between anchor <-> positive
    neg_dist = K.sum(K.abs(anchor_out - negative_out), axis=1)  # l1 dist between anchor <-> positive

    probs = K.softmax([pos_dist, neg_dist], axis=0)

    return K.mean(K.abs(probs[0]) + K.abs(1.0 - probs[1]))

def build_model(steps_per_epoch, epochs, b_size):
    input_layer = Input((resize[0], resize[1], 3))

    x = Conv2D(32, 3, activation="relu")(input_layer)
    x = Conv2D(32, 3, activation="relu")(x)
    x = MaxPool2D(2)(x)
    x = Conv2D(64, 3, activation="relu")(x)
    x = Conv2D(64, 3, activation="relu")(x)
    x = MaxPool2D(2)(x)
    x = Conv2D(128, 3, activation="relu")(x)
    x = Flatten()(x)
    x = Dense(100, activation="relu")(x)
    model = Model(input_layer, x)
    model.summary()

    triplet_model_a = Input((resize[0], resize[1], 3))
    triplet_model_p = Input((resize[0], resize[1], 3))
    triplet_model_n = Input((resize[0], resize[1], 3))

    triplet_model_out = Concatenate()([model(triplet_model_a), model(triplet_model_p), model(triplet_model_n)])
    triplet_model = Model([triplet_model_a, triplet_model_p, triplet_model_n], triplet_model_out)
    triplet_model.summary()

    triplet_model.compile(loss=triplet_loss, optimizer="adam", metrics=['accuracy', keras.metrics.BinaryAccuracy()])
    #input("Press Enter")

    training_data = ([anchors[0:int(len(anchors)*0.7)], positives[0:int(len(anchors)*0.7)], negatives[0:int(len(anchors)*0.7)]])
    testing_data = ([anchors[int(len(anchors)*0.7):], positives[int(len(anchors)*0.7):], negatives[int(len(anchors)*0.7):]])

    batch_size = int(len(anchors)*0.7)
    print(batch_size)

    #################UNCOMMENT###################

    history = triplet_model.fit(x=training_data, y=np.zeros((batch_size, 1)).astype("float32"), steps_per_epoch=steps_per_epoch, epochs=epochs, batch_size=b_size)

    y_test = np.zeros((int(len(anchors)) - (int(len(anchors) * 0.7)), 1)).astype("float32")

    evaluation = triplet_model.evaluate(testing_data, y_test, verbose=1)

    print(evaluation)
    #print(type(evaluation))
    try:
        loss, accuracy, f1_score, precision, recall = evaluation
        print("loss:", loss)
        print("accuracy:", accuracy)
        print("f1_score:", f1_score)
        print("precision:", precision)
        print("recall:", recall)
    except:
        print("could not unpack values")

    prediction = triplet_model.predict([anchors[:1], positives[:1], negatives[:1]])
    print("prediction shape:", prediction.shape)
    print(prediction)
    #x_test = x_test.reshape(x_test.shape[1],x_test.shape[2],x_test.shape[3],x_test.shape[4])
    #x_test = np.array(x_test[1], x_test[2], x_test[3], x_test[4])

    #model_embeddings = triplet_model.layers[3].predict(testing_data, verbose=1)
    #print(model_embeddings.shape)
    #import matplotlib.pyplot as plt

    #print("Accuracy Score:")
    #triplet_model.score(x_test, y_test)
    #y_pred = (prediction > 1)
    #matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    #print(matrix)
    #cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = matrix, display_labels = [False, True])
    #cm_display.plot()
    #plt.show()

    # summarize history for accuracy
    print(type(history))
    print(type(history.history))
    print(len(history.history))
    print(history.history)

    #plt.plot(history.history['loss'])
    #plt.plot(history.history['binary_accuracy'])
    #plt.title('model accuracy')
    #plt.ylabel('loss')
    #plt.xlabel('epoch')
    #plt.legend(['Loss', 'Binary_Accuracy'], loc='upper left')
    #plt.show()
    # summarize history for loss
    '''
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    '''
    '''
    reduced_embeddings = umap.UMAP(n_neighbors = 15, min_dist = 0.3, metric = 'correlaton').fit_transform(model_embeddings)
    print(reduced_embeddings.shape)
    
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:,1], c=y_test)'''
    return evaluation[0], evaluation[2], triplet_model


## Edit below function calling for image alteration

## CREATING INPUT IMAGE ARRAYS ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ ##

## Adjust these variables to alter input images ##
iterations = 100
simplified = False
contrast = 2.0  # Values for contrast are between 1.0 and 2.0 in gaps of .25
brightness = 0  # Values for brightness are between 0 and 100 in gaps of 20
(anchors, positives, negatives) = convertCSVtoImageArray(path=path, csvPath=csvPath, simplified=simplified, contrast=contrast, brightness=brightness)

start_time = time.time()
x_train = np.array((anchors[0:int(len(anchors)*0.7)], positives[0:int(len(anchors)*0.7)], negatives[0:int(len(anchors)*0.7):]))
x_test = np.array((anchors[int(len(anchors)*0.7):], positives[int(len(anchors)*0.7):], negatives[int(len(anchors)*0.7):]))
y_train = np.arange(0, int(len(anchors)*0.7))
y_test = np.arange(int(len(anchors)*0.7), int(len(anchors)))

print(anchors.shape)
print(positives.shape)
print(negatives.shape)
print(x_train.shape)
print(x_test.shape)

## CREATING INPUT IMAGE ARRAYS ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ ##

# main loop
avg_bin_acc = []
min_bin_acc = 100
max_bin_acc = 0
curr_acc = None
for i in range(0, iterations):
    # steps_per_epoch = 150, epochs = 50, b_size = 16
    print(i)
    _, curr_acc, t_model = build_model(steps_per_epoch=150, epochs=50, b_size=16)
    avg_bin_acc.append(curr_acc)

    if curr_acc<min_bin_acc:
        min_bin_acc=curr_acc

    if curr_acc>max_bin_acc:
        max_bin_acc=curr_acc
        curr_acc = round(curr_acc*100, 2)
        t_model.save("savedModels/TripletModel" + "Sim" + str(simplified) + "_" + str(contrast) + "_" + str(brightness) + "_" + str(curr_acc) + ".h5")
full_time = time.time() - start_time
print("Full Process Took:", full_time)
print("Averaging ", full_time/iterations, "per iteration")
print("Simplified:", simplified)
print("Contrast:", contrast)
print("Brightness:", brightness)
print("Binary Accuracy Array:", str(avg_bin_acc))
print("Average of binary accuracies: ", str(sum(avg_bin_acc) / len(avg_bin_acc)))
print("Lowest Model Accuracy:", min_bin_acc)
print("Highest Model Accuracy:", max_bin_acc)

# Below feedback arrays are not used
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
