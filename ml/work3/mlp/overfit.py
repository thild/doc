#!/usr/bin/python

import numpy as np
import pandas
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

X_train, y_train = load_svmlight_file('IMDBtrain.txt')
X_test, y_test = load_svmlight_file('IMDBtest.txt')

## save for the confusion matrix
label = y_test

## converts the labels to a categorical one-hot-vector
y_train = keras.utils.to_categorical(y_train, num_classes=2)
y_test = keras.utils.to_categorical(y_test, num_classes=2)



model = Sequential()
model.add(Dense(50, activation='relu', input_dim=100))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_split=0.10, epochs=320, batch_size=128, shuffle=True)

score = model.evaluate(X_test, y_test, batch_size=128)
print (score)

y_pred = model.predict_classes(X_test)
cm = confusion_matrix(label, y_pred)
print (cm)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy - 1 layer - 50 neurons - 320 epochs')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss - 1 layer - 50 neurons - 320 epochs')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

##################
