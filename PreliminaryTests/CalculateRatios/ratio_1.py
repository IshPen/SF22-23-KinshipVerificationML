import cv2
import numpy as np
import dlib
import math

#Horizontal Cross Section (0, 16) / Vertical Cross Section Average ((19, 6), (24, 10))/2

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
        left_side_larger = False
        print("LS:", left_side, "RS:", right_side)
        if left_side > right_side:
            left_side_larger = True
        return left_side_larger


    for face in faces:
        landmarks = predictor(img_gray, face)
        landmarks_points = []

        determine_long_side()

        ratio = linear_distance(0, 16)/((linear_distance(19, 6)+linear_distance(24, 10))/2)
        print(linear_distance(0, 16))
        print(linear_distance(19, 6))
        print(linear_distance(24, 10))
        print(ratio)

        return ratio

output = abs(main("face10.jpg")-main("face3.jpg"))
print(output)