#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:03:55 2019

@author: Pedro, Rodrigo, Alix
"""

import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# matplotlib inline
import os
from PIL import Image
import pathlib
import csv

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Keras
from keras import models
from keras import layers

import warnings
warnings.filterwarnings('ignore')

cmap = plt.get_cmap('inferno')

data = pd.read_csv('../data.csv')
data.head()
            
#data.shape

# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)

genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# len(y_train)

# len(y_test)

# X_train[10]

#model = models.Sequential()
#model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
#
#model.add(layers.Dense(128, activation='relu'))
#
#model.add(layers.Dense(64, activation='relu'))
#
#model.add(layers.Dense(10, activation='softmax'))
#
#model.compile(optimizer='adam',
#              loss='sparse_categorical_crossentropy',
#              metrics=['accuracy'])
#
#history = model.fit(X_train,
#                    y_train,
#                    epochs=20,
#                    batch_size=128)
#
#test_loss, test_acc = model.evaluate(X_test,y_test)
#
#print('test_acc: ',test_acc)

x_val = X_train[:200]
partial_x_train = X_train[200:]

y_val = y_train[:200]
partial_y_train = y_train[200:]

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=30,
                    batch_size=512,
                    validation_data=(x_val, y_val))
results = model.evaluate(X_test, y_test)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#predictions = model.predict(X_test)
#
#print(predictions[0].shape)
#
#print(np.sum(predictions[0]))
#
#print(np.argmax(predictions[0]))
