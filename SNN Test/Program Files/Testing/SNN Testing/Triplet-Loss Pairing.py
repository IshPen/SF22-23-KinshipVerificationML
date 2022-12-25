import pandas as pd
import random

fullFile = "E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\SNN Test\Data\list_of_pairs.csv"
relatedFile = "E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\SNN Test\Data\list_of_RELATED_pairs.csv"

fullDf = pd.read_csv(fullFile)
df = pd.read_csv(relatedFile)
listOfRelated = []


for i in df.values:
    listOfRelated.append([i[1], i[2]])

print("List of related pairs: ", listOfRelated)

def assignNPair(relatedArray):
    outputRelatedArray = []

    print(len(relatedArray))

    for i in range(0, len(relatedArray)):
        #chooses random image for n pair
        rand = random.choice(listOfRelated)[0]

        #makes sure random image is not the same as the related images
        if rand == relatedArray[i][0] or rand == relatedArray[i][1]:
            print("-----------------")
            print("Same image")
            print(relatedArray[i][0], relatedArray[i][1], rand)

            while rand == relatedArray[i][0] or rand == relatedArray[i][1]:
                print("reassigning")
                rand = random.choice(listOfRelated)[0]
                print(relatedArray[i][0], relatedArray[i][1], rand)



        #adds the n pair to the array
        outputRelatedArray.append([relatedArray[i][0], relatedArray[i][1], rand])
        #print("Related pair: ", relatedArray[i], " N pair: ", rand)
    return outputRelatedArray

pairs = assignNPair(listOfRelated)

print(pairs)

import csv
saveToFile = "E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\SNN Test\Data\Triplet-Loss-Pairs.csv"
#save 2d array to csv

with open(saveToFile, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(pairs)