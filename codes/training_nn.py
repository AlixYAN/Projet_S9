#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:03:55 2019

@author: Pedro, Rodrigo, Alix
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# matplotlib inline
import tensorflow as tf
import random

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix

# Keras
from keras import models
from keras import layers
from keras import optimizers

import warnings
warnings.filterwarnings('ignore')


#data = pd.read_csv('../data_all.csv')
data = pd.read_csv('../features_csv/data_reggae_hiphop.csv')

labels = np.unique(data['label'])
            
#data.shape

# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)

genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

conf_mat_total = []

#Seeds Initialization to reproduce results!
for seed_init in range(100):

    tf.set_random_seed(seed_init)
    np.random.seed(seed_init)
    random.seed(seed_init)
    
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

#    # Plot training & validation accuracy values
#    plt.plot(history.history['acc'])
#    plt.plot(history.history['val_acc'])
#    plt.title('Model accuracy')
#    plt.ylabel('Accuracy')
#    plt.xlabel('Epoch')
#    plt.legend(['Train', 'Test'], loc='upper left')
#    plt.show()
#    
#    # Plot training & validation loss values
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
#    plt.title('Model loss')
#    plt.ylabel('Loss')
#    plt.xlabel('Epoch')
#    plt.legend(['Train', 'Test'], loc='upper left')
#    plt.show()


    y_total = model.predict(X)
    
    y_total = np.argmax(y_total,axis=1)
    
    conf_mat_total.append(confusion_matrix(y_total,y))
    
    
    
# Calculate the mean and the variance of the confusion matrices
conf_mat_total_mean = np.mean(conf_mat_total,axis=0)
conf_mat_total_var = np.var(conf_mat_total,axis=0)

# We calculate the accuracy using the diagonal of the confusion matrix (the songs that were correctly classified)
accuracy_total_prediction = 100*np.trace(conf_mat_total_mean)/np.size(data,0)

# We save the confusion matrix as an CSV output
output_mean = pd.DataFrame(conf_mat_total_mean)
output_mean = output_mean.rename({idx:labels[idx] for idx in range(np.size(labels))},axis='columns')
output_mean = output_mean.rename({idx:labels[idx] for idx in range(np.size(labels))},axis='index')

output_mean.to_csv (r'../confusion_matrices/conf_matrix_mean_GMM.csv', index = True, header=True)

output_var = pd.DataFrame(conf_mat_total_var)
output_var = output_var.rename({idx:labels[idx] for idx in range(np.size(labels))},axis='columns')
output_var = output_var.rename({idx:labels[idx] for idx in range(np.size(labels))},axis='index')

output_var.to_csv (r'../confusion_matrices/conf_matrix_var_GMM.csv', index = True, header=True)

# Print the result to visualize in Spyder environement
print(output_mean)
print("Total accuracy = ", accuracy_total_prediction, "%")

#predictions = model.predict(X_test)
#
#print(predictions[0].shape)
#
#print(np.sum(predictions[0]))
#
#print(np.argmax(predictions[0]))
