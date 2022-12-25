import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

df = pd.read_csv(r"E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\BackupData\CleanedData\adj_cleaned_test_competition_ratios1-4.csv")

#dataset.head()
#print(df.isnull().sum())
df['r3']=df['r3'].fillna(df['r3'].mean())
#print(df.isnull().sum())

df.drop('index', axis=1, inplace=True)
df.drop('val', axis=1, inplace=True)
df.drop('primary_img', axis=1, inplace=True)
df.drop('secondary_img', axis=1, inplace=True)
#print(df.isnull().sum())

print(df.shape)

x = df.drop('set', 1)
y = df.set
'''seed = random.randint(0, 100)
print(seed)
'''

start = 702136
iter = 1000000

precision0 = []
precision1 = []
def run_model(seed):
  from sklearn.model_selection import train_test_split

  X_train, X_test, Y_train, Y_test= train_test_split(x, y, test_size= 0.15, random_state=seed, stratify = y)

  #Fitting the KNN model
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier(n_neighbors =15)
  knn.fit(X_train, Y_train)

  #Prediction of test set
  prediction_knn = knn.predict(X_test)

  #print(prediction_knn)
  #Print the predicted values
  #print("Prediction for test set: {}".format(prediction_knn))
  a = pd.DataFrame({'Actual value': Y_test, 'Predicted value': prediction_knn})
  a.head()

  #from sklearn import metrics
  from sklearn.metrics import classification_report, confusion_matrix

  #matrix = confusion_matrix(Y_test, prediction_knn)
  #sns.heatmap(matrix, annot=True, fmt="d")
  #plt.title('Confusion Matrix')
  #plt.xlabel('Predicted')
  #plt.ylabel('True')
  output = classification_report(Y_test, prediction_knn, output_dict = True)
  print(seed, output['0']['precision'], output['1']['precision'], str((seed/iter)*100))
  return [output['0']['precision'], output['1']['precision']]
  #outputdf = pd.DataFrame(output)

run_model(11)

file1 = open(r"E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\NeuralNetworkingTests\KNN\highestAccuracyKNNSeedRecord.txt","a")

for seed in range(start, iter):
  out = run_model(seed)
  precision0.append(out[0])
  precision1.append(out[1])


  file1.write("P0" + str(seed) + str(out[0]) + "\n")
  file1.write("P1" + str(seed) + str(out[1]) + "\n")
  #getting highest acc index and lowest acc index will be done in post-processing

file1.close()

#max precision0 index: [3883] 0.5412064570943076
#max precision1 index: [3883] 0.5600522193211488

#max precision0 index: [37437] 0.545897644191714
#max precision1 index: [37437] 0.56765899864682
#min precision0 index: [56546] 0.45829823083403537
#min precision1 index: [56546] 0.49605781865965837

precision0np = np.array(precision0)
precision1np = np.array(precision1)

indice0max = np.where(precision0np == np.amax(precision0np))
indice1max = np.where(precision1np == np.amax(precision1np))

indice0min = np.where(precision0np == np.amin(precision0np))
indice1min = np.where(precision1np == np.amin(precision1np))

print('max precision0 index:', indice0max[0], np.amax(precision0np))
print('max precision1 index:', indice1max[0], np.amax(precision1np))

print('min precision0 index:', indice0min[0], np.amin(precision0np))
print('min precision1 index:', indice1min[0], np.amin(precision1np))

file1 = open(r"E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\NeuralNetworkingTests\KNN\highestAccuracyKNNSeedRecord.txt", "a")
file1.write(str('max precision0 index:') + str(indice0max[0]) + str(np.amax(precision0np)) + "\n")
file1.write(str('max precision1 index:') + str(indice1max[0]) + str(np.amax(precision1np)) + "\n")
file1.close()

file2 = open(r"E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\NeuralNetworkingTests\KNN\highestAccuracyKNNSeedRecord.txt", "a")
file2.write(str('min precision0 index:') + str(indice0min[0]) + str(np.amin(precision0np)) + "\n")
file2.write(str('min precision1 index:') + str(indice1min[0]) + str(np.amin(precision1np)))
file2.close()