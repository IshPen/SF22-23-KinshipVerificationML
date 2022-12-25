import cv2
image = "face190.jpg"

img = cv2.imread(image)
hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

h,s,v = cv2.split(hsvImg)

hIsZero = True
sIsZero = True
vIsZero = True

for i in range(0, len(h)):
    if h[i].any() != 0:
        hIsZero = False
    if s[i].any() != 0:
        sIsZero = False
    if v[i].any() != 0:
        vIsZero = False

if hIsZero and sIsZero and vIsZero == False:
    print("Is Grayscale")
else:
    print('Is Color')