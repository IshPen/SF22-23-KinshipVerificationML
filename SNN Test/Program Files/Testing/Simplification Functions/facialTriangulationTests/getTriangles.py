# Derived from gColabDerivedSmoothApply.py
# This code should get all the triangles from a person's face
# and fill each one with a random color
import random

#RGB Differences
#Pairs:
#face221.jpg - 4500 1
#face221.jpg - 4584 1
#face221.jpg - 3 0
#face221.jpg - 47 0

import cv2
import numpy as np
import dlib
import math
#from google.colab.patches import cv2_imshow
facialTriangles = [(0, 2, 31), (2, 4, 31), (4, 6, 57), (6, 57, 10), (57, 12, 10), (35, 12, 14), (35, 14, 16)]
colorsList = [
(51, 183, 170),
(42, 204, 115),
(252, 212, 80),
(126, 21, 178),
(90, 221, 189),
(81, 232, 106),
(232, 92, 132),
(132, 0, 57),
(15, 129, 158),
(20, 5, 114),
(237, 225, 68),
(107, 206, 53),
(238, 244, 68),
(39, 8, 214),
(219, 10, 125)]
segmentImages = []*7
#f_root = "/content/drive/MyDrive/scienceFairFaces/"
f_root = "D:/Programs/Program Files/Pycharm Projects/SF22-23-KinshipVerificationML/Data/scienceFairRawFacesNOBG/"
'''
def multicolorMask(mask, output):
    cv2.imshow("mask_to_add", mask)
    #mask[np.all(mask==(255,255,255))] == random.choice[colorsList]
    output = cv2.bitwise_and(output, mask)
    cv2.imshow("output", output)
    cv2.waitKey()
    return output
'''
def facialTriangulation(p1, p2, p3, landmarks, img, img_gray):
    mask = np.zeros_like(img_gray)

    landmarks_points = []
    landmarks_points.append((landmarks.part(p1).x, landmarks.part(p1).y))
    landmarks_points.append((landmarks.part(p2).x, landmarks.part(p2).y))
    landmarks_points.append((landmarks.part(p3).x, landmarks.part(p3).y))

    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)

    # cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
    #cv2.imshow("mask", mask)
    #cv2.waitKey()
    cv2.fillConvexPoly(mask, convexhull, 255)
    #cv2.imshow("mask", mask)
    #cv2.waitKey()

    face_image_1 = cv2.bitwise_and(img, img, mask=mask)

    #face_image_1[np.all(face_image_1 != (0, 0, 0), axis=2)] = (255,255,255)

    return mask, face_image_1
def customAddImages(img1, img2, shade):
    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)
    completeImg = img1
    #shade = random.randint(124, 255)
    #completeImg = cv2.resize(completeImg, (completeImg.shape[0]*5, completeImg.shape[1]*5))
    for i in range(0, completeImg.shape[0]):
        for j in range(0, completeImg.shape[1]):
            if img2[i, j] != 0:
                completeImg[i, j] = shade
    return completeImg

def main_func(img):
  #init image
  img = cv2.imread(f_root+img)
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor("D:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\shape_predictor_68_face_landmarks.dat")
  faces = detector(img_gray)
  face = faces[0]

  landmarks = predictor(img_gray, face)

  output = np.zeros_like(img_gray)

  for i in facialTriangles:
    print(i)
    mask, face_image_1 = facialTriangulation(i[0], i[1], i[2], landmarks, img, img_gray)
    #cv2.imshow("mask", mask)
    #cv2.imshow("face_image_1", face_image_1)
    #cv2.waitKey()
    segmentImages.append(mask)

  finalOutput = segmentImages[0]
  for i in range(0, len(segmentImages)-1):
    finalOutput = customAddImages(finalOutput, segmentImages[i+1], 124+((255-124)/(i+1)))
    cv2.imshow("segment", segmentImages[i])
    cv2.imshow("finalOutput", finalOutput)
    cv2.waitKey()

  print("done")
  cv2.imshow("finalOutput", cv2.resize(finalOutput, (finalOutput.shape[0]*5, finalOutput.shape[1]*5)))

  #cv2.imshow("Image 1", img)
  img = cv2.resize(img, (img.shape[0]*4, img.shape[1]*4))
  face_image_1 = cv2.resize(face_image_1, (face_image_1.shape[0]*4, face_image_1.shape[1]*4))
  cv2.imshow("img", img)
  cv2.imshow("face_image_1", face_image_1)
  cv2.waitKey()
  #print(face_image_1)



  pixelColors = []

  for i in range(0, len(face_image_1)):
      for j in range(0, len(face_image_1[i])):
          if face_image_1[i][j].all() != 0:
              #makes sure it isn't black
              pixelColors.append(tuple(face_image_1[i][j]))
          #elif face_image_1[i][j].all() == 0:
              #print(face_image_1[i][j])

  bgrResult = average_tuple(pixelColors)
  print(round(average_tuple(pixelColors)[2]), round(average_tuple(pixelColors)[1]), round(average_tuple(pixelColors)[0]))
  return round(average_tuple(pixelColors)[2]), round(average_tuple(pixelColors)[1]), round(average_tuple(pixelColors)[0])


def average_tuple(nums):
    result = [sum(x) / len(x) for x in zip(*nums)]
    return result

def threeD_distance(x,y,z, x1,y1,z1):
  return math.sqrt((x-x1)**2 + (y-y1)**2 + (z-z1)**2)

out1 = main_func("noBGface3677.jpg")
out2 = main_func("noBGface2228.jpg")
print(threeD_distance(out1[0],out1[1],out1[2],out2[0],out2[1],out2[2]))
#print(midValAndSaturation(convertBGRtoHSV(bgrResult[0], bgrResult[1], bgrResult[2])))
#print(gVals)
#print(rVals)
#cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

#White 1 -
#Black 1 -