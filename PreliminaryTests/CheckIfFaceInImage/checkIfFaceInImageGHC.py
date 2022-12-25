#test one for github copilot

import cv2
import numpy as np
import dlib
import math

predictor = "E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
img = cv2.imread(r"E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\Data\scienceFairFaces\face188.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def find_face(img):
    faces = detector(img)
    if len(faces) == 0:
        return None
    return faces[0]

print(find_face(imgGray))