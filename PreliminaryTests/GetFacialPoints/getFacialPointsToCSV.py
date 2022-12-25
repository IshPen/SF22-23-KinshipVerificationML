import csv
import cmake
import cv2
import dlib
import os
from os import listdir
folder_dir = "E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\Data\scienceFairFaces"
imgList = []
dataToWrite = []
from imutils import face_utils
for image in os.listdir(folder_dir):
    imgList.append(image)

#print(imgList)
#print(len(imgList))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\shape_predictor_68_face_landmarks.dat")

def getLandmarks(img, rawImgName):
    frame = cv2.imread(img)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    xArray = []
    yArray = []
    outputArray = [rawImgName]
    faces = detector(gray)
    if len(faces) == 0:
        outputArray = [rawImgName, "No Face Found"]
    for face in faces:
        landmarks = predictor(gray, face)
        for points in range(0, 68):
            xArray.append(landmarks.part(points).x)
            yArray.append(landmarks.part(points).y)
    for i in range(0, len(xArray)):
        outputArray.append((xArray[i], yArray[i]))
    return outputArray

csvFile = 'E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\Data\FacialPoints.csv'
csvFileAlt = 'E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\Data\FacialPoints2.csv'

for i in range(0, len(imgList)):
    imageConcatenate = os.path.join(folder_dir, imgList[i])
    print(imgList[i])
    with open(csvFile, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(getLandmarks(imageConcatenate, imgList[i]))

with open(csvFile) as input, open(csvFileAlt, 'w', newline='') as output:
    writer = csv.writer(output)
    for row in csv.reader(input):
        if any(field.strip() for field in row):
            writer.writerow(row)