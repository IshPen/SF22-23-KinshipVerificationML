import cv2
import pandas as pd
import numpy as np

df = pd.read_csv(r'E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\NeuralNetworkingTests\Datasets\adj_cleaned_test_competition_ratios1-4.csv')
df = df.head(100)
#print(data)
df = df.reset_index()  # make sure indexes pair with number of rows
X = []
output = []
target=df.set

def addToX(img1, img2):
  primary_img = cv2.imread(r"E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\Data\scienceFairRawFacesNOBG\noBG" + img1)
  secondary_img = cv2.imread(r"E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\Data\scienceFairRawFacesNOBG\noBG" + img2)
  dsize = (108, 124)

  print(img1 + img2)
  #resize img if not 108x124 resolution
  if primary_img.shape[0] != 108 or primary_img.shape[1] != 124:
    primary_img = cv2.resize(primary_img, dsize)
  if secondary_img.shape[0] != 108 or secondary_img.shape[1] != 124:
    secondary_img = cv2.resize(secondary_img, dsize)

  #return np.concatenate((primary_img, secondary_img))
  return primary_img, secondary_img



for index, row in df.iterrows():
    #print(row['primary_img'], row['secondary_img'])
    output = addToX(row['primary_img'], row['secondary_img'])
    X.append(output)
    #print(output)

Xsaved = X
X = np.array(X)
print(X.shape)

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
for i in range(0, 20):
  X_train, X_test, y_train, y_test = train_test_split(X,target,test_size=0.1)
  print(X_test.shape)
  print(X_train.shape)
  X_train = (X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3])).tolist()
  #print(X_train.shape)
  print(X_train)

  model = GaussianNB()
  model.fit(X_train,y_train)
  print(model.score(X_test,y_test))