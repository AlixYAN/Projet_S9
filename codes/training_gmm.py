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
import matplotlib as mpl
import random
# matplotlib inline
import os
from PIL import Image
import pathlib
import csv

from sklearn import datasets
from sklearn.externals.six.moves import xrange
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

#cmap = plt.get_cmap('inferno')

data = pd.read_csv('../data_feat.csv')
data.head()
            
#data.shape

#Initialization of seeds to reproduce results
seed_init = 100
np.random.seed(seed_init)
random.seed(seed_init)

# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)

genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#x_val = X_train[:200]
#partial_x_train = X_train[200:]
#
#y_val = y_train[:200]
#partial_y_train = y_train[200:]

n_classes = len(np.unique(y_train))

# Try GMMs using different types of covariances.
estimators = {cov_type: GaussianMixture(n_components=n_classes,
              covariance_type=cov_type, max_iter=20, random_state=0)
              for cov_type in ['spherical', 'diag', 'full']}

n_estimators = len(estimators)

for index, (name, estimator) in enumerate(estimators.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    estimator.means_init = np.array([X_train[y_train == i].mean(axis=0)
                                    for i in range(n_classes)])

    # Train the other parameters using the EM algorithm.
    estimator.fit(X_train)

    genres = np.unique(genre_list)

    y_train_pred = estimator.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100

    y_test_pred = estimator.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100

    conf_mat = 100*confusion_matrix(y_test,y_test_pred)/(0.2*np.size(data,0))

    print("Confusion Matrix with cov =", estimator.covariance_type ,"in %\n", conf_mat)

    total_acc = np.trace(conf_mat)

    print("Test accuracy with cov =", estimator.covariance_type ,"\n" , total_acc , "%")