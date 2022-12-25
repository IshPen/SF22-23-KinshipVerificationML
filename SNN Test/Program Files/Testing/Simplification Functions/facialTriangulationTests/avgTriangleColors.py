# Order of Processes -
# 1. Get color triangle from face_image_1
# 2. Get avg color of triangle by looking at all non black triangles
# 3. Replace all coordinates in the triangle with avg color
# 4. Show each color triangle
import random

import cv2
import numpy as np
import dlib
import math
#from google.colab.patches import cv2_imshow
facialTriangles = [(0, 2, 31), (2, 4, 31), (4, 6, 57), (6, 57, 10), (57, 12, 10), (35, 12, 14), (35, 14, 16), (0, 28, 30), (0, 30, 31), (28, 30, 16), (16, 30, 35), (6, 8, 10), (48, 51, 57), (51, 54, 57)]
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
#f_root = "/content/drive/MyDrive/scienceFairFaces/"
f_root = "D:/Programs/Program Files/Pycharm Projects/SF22-23-KinshipVerificationML/Data/scienceFairRawFacesNOBG/"

def colorsInTriangle(img):
    pixelColors = []
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            if img[i][j].all() != 0:
                # makes sure it isn't black
                pixelColors.append(tuple(img[i][j]))
    return pixelColors

def average_tuple(nums):
    result = [sum(x) / len(x) for x in zip(*nums)]
    return result

def getTriangle(p1, p2, p3, landmarks, img_gray, img):
    landmarks_points = []
    mask = np.zeros_like(img_gray)
    landmarks_points.append((landmarks.part(p1).x, landmarks.part(p1).y))
    landmarks_points.append((landmarks.part(p2).x, landmarks.part(p2).y))
    landmarks_points.append((landmarks.part(p3).x, landmarks.part(p3).y))
    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)
    # cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
    cv2.fillConvexPoly(mask, convexhull, 255)
    triangle_img = cv2.bitwise_and(img, img, mask=mask)
    return triangle_img

def writeAvgColorToTriangle(avgColor, triangle):
    for i in range(0, len(triangle)):
        for j in range(0, len(triangle[i])):
            if triangle[i, j].all() != 0:
                triangle[i, j] = avgColor
    return triangle

def customAddColorImages(img1, img2):
    #cv2.imshow("img1", img1)
    #cv2.imshow("img2", img2)
    completeImg = img1
    for i in range(0, completeImg.shape[0]):
        for j in range(0, completeImg.shape[1]):
            if img2[i, j].all() != 0:
                completeImg[i, j] = img2[i, j]
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

  segmentImages = []
  for i in range(0, len(facialTriangles)):
      triangle_img = getTriangle(facialTriangles[i][0], facialTriangles[i][1], facialTriangles[i][2], landmarks, img_gray, img)
      pixelsList = colorsInTriangle(triangle_img)
      bgrResult = average_tuple(pixelsList)
      rgbResult = (bgrResult[2], bgrResult[1], bgrResult[0])
      print(rgbResult)
      #triangle_img = cv2.resize(triangle_img, (triangle_img.shape[0] * 5, triangle_img.shape[1] * 5))
      #cv2.imshow("triangle_img", triangle_img)
      #cv2.waitKey()
      avgColorTriangle = writeAvgColorToTriangle(bgrResult, triangle_img)
      segmentImages.append(avgColorTriangle)
      #cv2.imshow("avgColorTriangle", avgColorTriangle)
      #cv2.waitKey()

  finalOutput = segmentImages[0]
  for i in range(0, len(segmentImages) - 1):
      finalOutput = customAddColorImages(finalOutput, segmentImages[i + 1])
      #cv2.imshow("segment", segmentImages[i])
      #cv2.imshow("finalOutput", finalOutput)
      #cv2.waitKey()


  finalOutput = customAddColorImages(img, finalOutput)
  finalOutput = cv2.resize(finalOutput, (finalOutput.shape[0]*5, finalOutput.shape[1]*5))
  cv2.imshow("finalOutput", finalOutput)
  cv2.waitKey()

  #finalOutput = cv2.GaussianBlur(finalOutput,(25,25),cv2.BORDER_DEFAULT)
  #cv2.imshow("finalOutput", finalOutput)
  #cv2.waitKey()
  rgbVals = round(average_tuple(pixelsList)[2]), round(average_tuple(pixelsList)[1]), round(average_tuple(pixelsList)[0])
  print("Final AVG RGB: ", rgbVals)
  return rgbVals


def threeD_distance(x,y,z, x1,y1,z1):
  return math.sqrt((x-x1)**2 + (y-y1)**2 + (z-z1)**2)

if __name__ == "__main__":
    out1 = main_func("noBGface9.jpg")
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