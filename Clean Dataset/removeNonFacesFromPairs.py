import csv

faces = r"E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\Data\FacialPoints2.csv"
test_competition = r"E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\Data\test_competition.csv"
list = []
pairs_list = []
counter = 0


def copy_csv(filename):
    import pandas as pd
    df = pd.read_csv(filename)
    df.to_csv('copy_of_' + 'test_competition.csv')


copy_csv(test_competition)

with open(faces, 'r') as csvfile:
    reader = csv.reader(open(faces))
    for i in reader:
        if 'No Face Found' not in i:
            #print(i)
            list.append(i[0])

#list contains all images with a face
print(len(list))

reader = csv.reader(open(test_competition))
print(reader)
for i in reader:
    #print(i)
    if i[1] in list and i[2] in list:
        #print(i)
        counter +=1
        pairs_list.append(i)

print(counter/2)
#pairs_list contains all image pairs with both faces
#print(pairs_list)
print("len of pairs_list: ", len(pairs_list))

pairs_list_post_relations = []
for i in pairs_list:
    if 'md' in i or 'ms' in i or 'fd' in i or 'fs' in i:
        pairs_list_post_relations.append(i)

#pairs_list_pr contains all valid image pairs that are of parent and child
#for i in pairs_list_post_relations:
    #print(i)

#print(pairs_list)
#print(len(pairs_list)/2)
#print(pairs_list_post_relations)
#print(len(pairs_list_post_relations)/2)

with open("cleaned_test_competition.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerows(pairs_list_post_relations)