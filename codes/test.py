#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 09:02:37 2020

@author: Pedro
"""
import os
from classification_nn import classify_nn
genres = ['hiphop']
for g in genres:
    for filename in os.listdir(f'../dataset/{g}'):
        songname = f'../dataset/{g}/{filename}'
        c = classify_nn(songname)
        print(c)