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
import tensorflow as tf
import random
# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Keras
from keras import models
from keras import layers
from keras import optimizers

import warnings
warnings.filterwarnings('ignore')

#Seeds Initialization to reproduce results!
seed_init = 20
tf.set_random_seed(seed_init)
np.random.seed(seed_init)
random.seed(seed_init)

data = pd.read_csv('../data_feat.csv')
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

x_val = X_train[:int(0.2*np.size(data,0))]
partial_x_train = X_train[int(0.2*np.size(data,0)):]

y_val = y_train[:int(0.2*np.size(data,0))]
partial_y_train = y_train[int(0.2*np.size(data,0)):]

n_classes = len(np.unique(y_train))

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
#model.add(layers.Dense(256, activation='relu'))
#model.add(layers.Dense(128, activation='relu'))
#model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(n_classes, activation='softmax'))

opt = optimizers.Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)

model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=32,
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
