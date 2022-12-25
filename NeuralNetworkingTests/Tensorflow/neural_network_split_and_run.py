import pandas as pd
import tensorflow as tf
import numpy as np
import winsound

df = pd.read_csv(r"E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\NeuralNetworkingTests\Datasets\adj_cleaned_test_competition_ratios1-4.csv")
#print(df.head())

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.15, random_state = 42)

print((train))
print((test))

train = pd.DataFrame(train)
test = pd.DataFrame(test)

y_train = list(train['set'])
print(y_train)

x_train = train
x_train.drop('set', axis=1, inplace=True)
x_train.drop('index', axis=1, inplace=True)
x_train.drop('val', axis=1, inplace=True)
x_train.drop('primary_img', axis=1, inplace=True)
x_train.drop('secondary_img', axis=1, inplace=True)
print(x_train)
x_train = x_train/255.0

y_test = list(test['set'])
print(y_test)

x_test = test
x_test.drop('set', axis=1, inplace=True)
x_test.drop('index', axis=1, inplace=True)
x_test.drop('val', axis=1, inplace=True)
x_test.drop('primary_img', axis=1, inplace=True)
x_test.drop('secondary_img', axis=1, inplace=True)
print(x_test)
x_test = x_test/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2,)
])

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss,
              metrics=['accuracy'])


x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)
winsound.Beep(440, 3000)