import csv
import cv2
import dlib
import math
import os
import numpy as np

import time
start = time.time()
#Horizontal Cross Section (0, 16) / Vertical Cross Section Average ((19, 6), (24, 10))/2

def ratio_1(faceInput):
    faceInput = os.path.join("E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\Data\scienceFairFaces", faceInput)
    img = cv2.imread(faceInput)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #mask = np.zeros_like(img_gray)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("E:\Programs\Program Files\Pycharm "
                                     "Projects\SF22-23-KinshipVerificationML\shape_predictor_68_face_landmarks.dat")
    faces = detector(img_gray)


    def linear_distance(point_a, point_b):
        return math.sqrt((landmarks.part(point_a).y - landmarks.part(point_b).y) ** 2 + (landmarks.part(point_a).x - landmarks.part(point_b).x) ** 2)


    def determine_long_side():
        left_side = linear_distance(2, 31)
        right_side = linear_distance(14, 35)
        left_side_larger = False
        #print("LS:", left_side, "RS:", right_side)
        if left_side > right_side:
            left_side_larger = True
        return left_side_larger


    for face in faces:
        landmarks = predictor(img_gray, face)
        landmarks_points = []

        determine_long_side()

        ratio = linear_distance(0, 16)/((linear_distance(19, 6)+linear_distance(24, 10))/2)
        #print(linear_distance(0, 16))
        #print(linear_distance(19, 6))
        #print(linear_distance(24, 10))
        #print(ratio)

        return ratio

def ratio_2(faceInput):
    faceInput = os.path.join("E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\Data\scienceFairFaces", faceInput)
    img = cv2.imread(faceInput)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #mask = np.zeros_like(img_gray)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("E:\Programs\Program Files\Pycharm "
                                     "Projects\SF22-23-KinshipVerificationML\shape_predictor_68_face_landmarks.dat")
    faces = detector(img_gray)


    def linear_distance(point_a, point_b):
        return math.sqrt((landmarks.part(point_a).y - landmarks.part(point_b).y) ** 2 + (landmarks.part(point_a).x - landmarks.part(point_b).x) ** 2)


    def determine_long_side():
        left_side = linear_distance(2, 31)
        right_side = linear_distance(14, 35)
        larger_side = False
        #print("LS:", left_side, "RS:", right_side)
        if left_side > right_side:
            larger_side = "left"
        elif right_side >= left_side:
            larger_side = "right"
        return larger_side


    for face in faces:
        landmarks = predictor(img_gray, face)
        landmarks_points = []

        longer_side = determine_long_side()
        if longer_side == "left":
            ratio = linear_distance(6, 41)/(linear_distance(6, 19))
            #print(linear_distance(6, 41))
            #print(linear_distance(19, 6))
        elif longer_side == "right":
            ratio = linear_distance(10, 46)/linear_distance(10, 24)
            #print(linear_distance(10, 46))
            #print(linear_distance(10, 24))

        #print(ratio)
    return ratio

def ratio_3(faceInput):
    faceInput = os.path.join("E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\Data\scienceFairFaces", faceInput)
    img = cv2.imread(faceInput)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("E:\Programs\Program Files\Pycharm "
                                     "Projects\SF22-23-KinshipVerificationML\shape_predictor_68_face_landmarks.dat")
    faces = detector(img_gray)

    for face in faces:
        landmarks = predictor(img_gray, face)
        landmarks_points = []
        '''for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)'''
        # landmarks_points.append((x, y))
        # sqrt of left side math.sqrt((y2-y1)^2 +(x2-x1)^2)
        # cv2.line(img, (landmarks.part(2).x, landmarks.part(2).y), (landmarks.part(31).x, landmarks.part(31).y), (0, 0, 255), 2)
        # cv2.line(img, (landmarks.part(14).x, landmarks.part(14).y), (landmarks.part(35).x, landmarks.part(35).y), (0, 0, 255), 2)

        left_side = math.sqrt(
            (landmarks.part(2).y - landmarks.part(31).y) ** 2 + (landmarks.part(2).x - landmarks.part(31).x) ** 2)
        right_side = math.sqrt(
            (landmarks.part(14).y - landmarks.part(35).y) ** 2 + (landmarks.part(14).x - landmarks.part(35).x) ** 2)

        #print("LS:", left_side, "RS:", right_side)
        if left_side > right_side:
            landmarks_points.append((landmarks.part(2).x, landmarks.part(2).y))
            landmarks_points.append((landmarks.part(4).x, landmarks.part(4).y))
            landmarks_points.append((landmarks.part(32).x, landmarks.part(32).y))
        else:
            landmarks_points.append((landmarks.part(14).x, landmarks.part(14).y))
            landmarks_points.append((landmarks.part(16).x, landmarks.part(16).y))
            landmarks_points.append((landmarks.part(36).x, landmarks.part(36).y))

        points = np.array(landmarks_points, np.int32)
        convexhull = cv2.convexHull(points)
        # cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
        cv2.fillConvexPoly(mask, convexhull, 255)
        face_image_1 = cv2.bitwise_and(img, img, mask=mask)

    # cv2.imshow("Image 1", img)
    #cv2.imshow("Face image 1", face_image_1)
    # print(face_image_1)

    pixelColors = []

    for i in range(0, len(face_image_1)):
        for j in range(0, len(face_image_1[i])):
            if face_image_1[i][j].all() != 0:
                # makes sure it isn't black
                pixelColors.append(tuple(face_image_1[i][j]))
            #elif face_image_1[i][j].all() == 0:
                #print(face_image_1[i][j])

    def average_tuple(nums):
        result = [sum(x) / len(x) for x in zip(*nums)]
        return result

    rgbResult = average_tuple(pixelColors)
    #print(rgbResult)

    def convertBGRtoHSV(B, G, R):
        # Constraining the values to the range 0 to 1
        R_dash = R / 255
        G_dash = G / 255
        B_dash = B / 255
        # defining the following terms for convenience
        Cmax = max(R_dash, G_dash, B_dash)
        Cmin = min(R_dash, G_dash, B_dash)
        delta = Cmax - Cmin
        # hue calculation
        if (delta == 0):
            H = 0
        elif (Cmax == R_dash):
            H = (60 * (((G_dash - B_dash) / delta) % 6))
        elif (Cmax == G_dash):
            H = (60 * (((B_dash - R_dash) / delta) + 2))
        elif (Cmax == B_dash):
            H = (60 * (((R_dash - G_dash) / delta) + 4))

        # saturation calculation
        if (Cmax == 0):
            S = 0
        else:
            S = delta / Cmax
        # value calculation
        V = Cmax
        # print output. H in degrees. S and V in percentage.
        # these values may also be represented from 0 to 255.
        #print("H = {:.1f}°".format(H))
        #print("S = {:.1f}%".format(S * 100))
        #print("V = {:.1f}%".format(V * 100))
        hsvTuple = (H, S, V)
        return hsvTuple

    # print(convertBGRtoHSV(rgbResult[0], rgbResult[1], rgbResult[2]))

    def midValAndSaturation(hsvTuple):
        H = hsvTuple[0]
        S = (hsvTuple[1]) * 100
        V = (hsvTuple[2]) * 100
        #print(H, S, V)
        H = (H * 75) / V
        #print(H)
        H = ((H * 50) / S) / 2
        #print(H)
        return H
    #print(midValAndSaturation(convertBGRtoHSV(rgbResult[0], rgbResult[1], rgbResult[2])))
    return (midValAndSaturation(convertBGRtoHSV(rgbResult[0], rgbResult[1], rgbResult[2])))

def ratio_4(faceInput):
    faceInput = os.path.join("E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\Data\scienceFairFaces", faceInput)
    img = cv2.imread(faceInput)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #mask = np.zeros_like(img_gray)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("E:\Programs\Program Files\Pycharm "
                                     "Projects\SF22-23-KinshipVerificationML\shape_predictor_68_face_landmarks.dat")
    faces = detector(img_gray)


    def linear_distance(point_a, point_b):
        return math.sqrt((landmarks.part(point_a).y - landmarks.part(point_b).y) ** 2 + (landmarks.part(point_a).x - landmarks.part(point_b).x) ** 2)

    for face in faces:
        landmarks = predictor(img_gray, face)
        landmarks_points = []

        #longer_side = determine_long_side()
        '''
        if longer_side == "left":
            ratio = linear_distance(6, 41)/(linear_distance(6, 19))
            print(linear_distance(6, 41))
            print(linear_distance(19, 6))
        elif longer_side == "right":
            ratio = linear_distance(10, 46)/linear_distance(10, 24)
            print(linear_distance(10, 46))
            print(linear_distance(10, 24))
        '''
        ratio = linear_distance(8, 51)/linear_distance(8, 27)
        #print(ratio)
    return ratio


reader = csv.reader(open('E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\BackupData\CleanedData\cleaned_test_competition_ratios1-3.csv', 'r'))
columnToWrite = []


for i in reader:
    #print(i)
    #extra code ([2:-1]) was added to remove strings from face file paths, need to find a more permanent solution,
    #probably should remove all strings from csv file

    #ratio_3_output = abs(ratio_3(i[1]) - ratio_3(i[2]))
    #i.append(ratio_3_output)

    #ratio_1_output = abs(ratio_1(i[1]) - ratio_1(i[2]))*100
    #i.append(ratio_1_output)
    #ratio_2_output = abs(ratio_2(i[1]) - ratio_2(i[2]))*100
    #i.append(ratio_2_output)

    '''
    ratio_3_output = abs(ratio_3(i[1]) - ratio_3(i[2]))
    i.append(ratio_3_output)

    '''
    ratio_4_output = abs(ratio_4(i[1]) - ratio_4(i[2]))*100
    i.append(ratio_4_output)

    print(i)
    columnToWrite.append(i)

writer = csv.writer(open('uncleaned_test_competition_ratios1-4.csv', 'w'))

for i in columnToWrite:
    writer.writerow([i])


end = time.time()
print("Total time:", end-start)