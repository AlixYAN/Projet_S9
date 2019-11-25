#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 09:09:46 2019

@author: Pedro, Rodrigo, Alix
"""
import pandas as pd
import numpy as np
import os
import essentia.standard as es

# Read the csv file which you want to add a new feature to avoid recalculation 
# of all the parameters
data = pd.read_csv('../features_csv/data_reagge_hiphop.csv')

data_new = data

# Put the feature you want to delete here!
del_feat = ["zcr"]
        
data_new = data_new.drop(del_feat,axis=1)