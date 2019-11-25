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
data_new = pd.read_csv('../features_csv/data_selected.csv')

genres = data_new['label']
songs = data_new['filename']

sr = 22050

#Constructor of the new feature extractor Essentia


for i in np.arange(np.size(data_new,0)):
    g = genres[i]
    filename = songs[i]
    songname = f'../dataset/{g}/{filename}'
    audio = es.MonoLoader(filename=songname,sampleRate=sr)()
        
    # Put the feature you want to add here!

data_new.insert(np.size(data_new,1)-1,"name",name,True)

data_new.to_csv(r'../features_csv/data_selected_2.csv', index = False, header=True)
