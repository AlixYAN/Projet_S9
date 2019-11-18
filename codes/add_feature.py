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

genres = data['label']
songs = data['filename']

sr = 22050

#Constructor of the extractor Essentia
zcr_extractor = es.ZeroCrossingRate()
zcr = []

for i in np.arange(np.size(data,0)):
    g = genres[i]
    filename = songs[i]
    songname = f'../dataset/{g}/{filename}'
    audio = es.MonoLoader(filename=songname,sampleRate=sr)()
        
    # Put the feature you want to add here!
    zcr.append(zcr_extractor(audio))
        
data_new.insert(np.size(data_new,1)-1,"Zcr",zcr,True)