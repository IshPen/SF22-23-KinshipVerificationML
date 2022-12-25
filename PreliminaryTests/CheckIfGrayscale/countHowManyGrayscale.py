import cv2
import csv
import os
test_competition = "FacialPoints2.csv"
listOfGrayscales = []
counter = 0

def checkIfGrayscale(image):
    image = os.path.join("E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\Data\scienceFairFaces", image)
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
        #print("Is Grayscale")
        return True
    else:
        #print('Is Color')
        return False



with open(test_competition, 'r') as csvfile:
    reader = csv.reader(open(test_competition))
    for i in reader:
        if checkIfGrayscale(i[0]) and i[1] != "No Face Found":
            #print(i)
            listOfGrayscales.append(i[0])
            counter +=1
            print(counter, i)

#list contains all images with an image in grayscale
print(len(listOfGrayscales))

'''reader = csv.reader(open(test_competition))
print(reader)
pairs_list = []

for i in reader:
    #print(i)
    if i[1] in listOfGrayscales:
        #print(i)
        counter +=1
        pairs_list.append(i)

'''
non_grayscale_list = []
reader = csv.reader(open("cleaned_test_competition2.csv"))
for i in reader:
    if i[1] not in listOfGrayscales and i[2] not in listOfGrayscales:
        non_grayscale_list.append(i)

print(counter/2)
#pairs_list contains all image pairs with both faces
#print(pairs_list)
print("len of pairs_list: ", len(non_grayscale_list))

#pairs_list_pr contains all valid image pairs that are of parent and child
#for i in pairs_list_post_relations:
    #print(i)

#print(pairs_list)
#print(len(pairs_list)/2)
#print(pairs_list_post_relations)
#print(len(pairs_list_post_relations)/2)

with open("cleaned_test_competition3.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerows(non_grayscale_list)