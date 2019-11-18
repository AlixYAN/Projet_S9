#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:03:55 2019

@author: Pedro, Rodrigo, Alix
"""

import pandas as pd
import numpy as np
import random
# matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

#cmap = plt.get_cmap('inferno')

data = pd.read_csv('../features_csv/data_all.csv')
#data = pd.read_csv('../features_csv/data_reagge_hiphop.csv')
           
labels = np.unique(data['label'])
 
#data.shape

#Initialization of seeds to reproduce results
seed_init = 1
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
              for cov_type in ['full']}

#Spherical = variance is the same along all axes, which is not true;
#Tied = All gaussians share the same covariance matrix, which is not true;
#Diagonal = Assumes the features are independent from each other, may be interesting
#Full = Everyone has it's own covariance matrix

n_estimators = len(estimators)

for index, (name, estimator) in enumerate(estimators.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    estimator.means_init = np.array([X_train[y_train == i].mean(axis=0)
                                    for i in range(n_classes)])

    # Train the other parameters using the EM algorithm.
    estimator.fit(X_train)

    y_train_pred = estimator.predict(X_train)
    accuracy_train = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100

    y_test_pred = estimator.predict(X_test)
    accuracy_test = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100   
    
    
    # We apply now the trained model to the whole data to see how it behaves
    y_total = estimator.predict(X)
    
    # To visualize the result, we have a confusion matrix
    conf_mat_total = confusion_matrix(y_total,y)
    
    # We calculate the accuracy using the diagonal of the confusion matrix (the songs that were correctly classified)
    accuracy_total_prediction = 100*np.trace(conf_mat_total)/np.size(data,0)
    
    # We save the confusion matrix as an CSV output
    output = pd.DataFrame(conf_mat_total)
    output = output.rename({idx:labels[idx] for idx in range(np.size(labels))},axis='columns')
    output = output.rename({idx:labels[idx] for idx in range(np.size(labels))},axis='index')
    
    output.to_csv (r'../confusion_matrices/conf_matrix_GMM.csv', index = True, header=True)
    
    # Print the result to visualize in Spyder environement
    print(output)
    print("Total accuracy = ", accuracy_total_prediction, "%")