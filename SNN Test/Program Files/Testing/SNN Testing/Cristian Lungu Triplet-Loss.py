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
        a_i = cv2.resize(a_i, (108, 124))

        p_i = cv2.imread(str(path + r"\noBG" + positive_paths[image]))
        p_i = cv2.resize(p_i, (108, 124))

        n_i = cv2.imread(str(path + r"\noBG" + negative_paths[image]))
        n_i = cv2.resize(n_i, (108, 124))

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
