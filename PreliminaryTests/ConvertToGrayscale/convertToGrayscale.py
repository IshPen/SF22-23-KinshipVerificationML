import cv2
file_name = 'face1.jpg'
img = cv2.imread(file_name)
gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imwrite("opencv-greyscale.png",gray_image)

cv2.imshow("Original", img)
cv2.imshow("Grayscale", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()