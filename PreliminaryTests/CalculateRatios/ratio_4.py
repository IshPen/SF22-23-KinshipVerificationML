import cv2
import numpy as np
import dlib
import math

#max side(length from eye to jaw 8, 51)/full length from chin to above eye 8, 27

def main(faceInput):
    img = cv2.imread(faceInput)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
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
        print("LS:", left_side, "RS:", right_side)
        if left_side > right_side:
            larger_side = "left"
        elif right_side >= left_side:
            larger_side = "right"
        return larger_side


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
        print(ratio)
    return ratio

output = abs(main("face11.jpg")-main("face279.jpg"))
print(output)