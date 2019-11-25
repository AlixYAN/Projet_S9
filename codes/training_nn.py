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
from sklearn.metrics import confusion_matrix
# Keras
from keras import models
from keras import layers

import warnings
warnings.filterwarnings('ignore')

cmap = plt.get_cmap('inferno')

data = pd.read_csv('./data.csv')
data.head()
            
labels = np.unique(data['label'])

#data.shape

# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)
#data = data.drop(['chroma_stft'],axis=1)
data = data.drop(['rmse'],axis=1)
#data = data.drop(['spectral_centroid'],axis=1)
data = data.drop(['spectral_bandwidth'],axis=1)
data = data.drop(['rolloff'],axis=1)
data = data.drop(['zero_crossing_rate'],axis=1)
data = data.drop(['tempo'],axis=1)
data = data.drop(['utempo'],axis=1)

#{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} {np.mean(tempo)} {np.mean(utempo)}'

genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)


for i in range(10) :

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
                        epochs=50,
                        batch_size=512,
                        validation_data=(x_val, y_val))
    
    results = model.evaluate(X_test, y_test)
    
    # Plot training & validation accuracy values
    #plt.plot(history.history['acc'])
    #plt.plot(history.history['val_acc'])
    #plt.title('Model accuracy')
    #plt.ylabel('Accuracy')
    #plt.xlabel('Epoch')
    #plt.legend(['Train', 'Test'], loc='upper left')
    #plt.show()
    
    # Plot training & validation loss values
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.title('Model loss')
    #plt.ylabel('Loss')
    #plt.xlabel('Epoch')
    #plt.legend(['Train', 'Test'], loc='upper left')
    #plt.show()
    
    
    y_total = model.predict(X)
    
    y_total = np.argmax(y_total,axis=1)
    
    conf_mat = confusion_matrix(y_total,y)
    
    if i == 0:
        conf_mat_total = conf_mat 
    else:
        for j in range(4):
            for k in range(4):
                 conf_mat_total[j][k] = float(conf_mat_total[j][k]*i + conf_mat[j][k])/float(i+1)
    
    total_acc = np.trace(conf_mat_total)
        
    output = pd.DataFrame(conf_mat_total)
    output = output.rename({idx:labels[idx] for idx in range(np.size(labels))},axis='columns')
    output = output.rename({idx:labels[idx] for idx in range(np.size(labels))},axis='index')
    
    export_csv = output.to_csv (r'../conf_matrix_FFNN.csv', index = True, header=True)


    for clabel in range(5) :
        plt.plot(data['chroma_stft'][clabel*100:99+clabel*100], data['spectral_centroid'][clabel*100:99+clabel*100], 'x')
        plt.ylabel('Spec Cent')
        plt.xlabel('Chroma STFT')
    plt.legend(['Classical', 'HipHop', 'Jazz', 'Reggae', 'Rock'], loc='upper left')
    plt.show()


#predictions = model.predict(X_test)
#
#print(predictions[0].shape)
#
#print(np.sum(predictions[0]))
#
#print(np.argmax(predictions[0]))

#predictions = model.predict(X_test)
#
#print(predictions[0].shape)
#
#print(np.sum(predictions[0]))
#
#print(np.argmax(predictions[0]))