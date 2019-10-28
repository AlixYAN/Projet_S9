#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 09:09:46 2019

@author: Pedro, Rodrigo, Alix
"""

# feature extractoring and preprocessing data
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# matplotlib inline
import os
from PIL import Image
import pathlib
import csv
import scipy.stats

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Keras
from keras import models
from keras import layers

import warnings
warnings.filterwarnings('ignore')

cmap = plt.get_cmap('inferno')

#plt.figure(figsize=(10,10))
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
#for g in genres:
#    pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)     
#    for filename in os.listdir(f'../dataset/{g}'):
#        songname = f'../dataset/{g}/{filename}'
#        y, sr = librosa.load(songname, mono=True, duration=5)
#        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
#        plt.axis('off');
#        plt.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png')
#        plt.clf()
#        
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate tempo utempo'
n_mfcc = 5
for i in range(1, n_mfcc+1):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

file = open('../data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    for filename in os.listdir(f'../dataset/{g}'):
        songname = f'../dataset/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#       Add TEMPO features
        onset_env = librosa.onset.onset_strength(y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        tempo = tempo.item()
        prior = scipy.stats.uniform(40, 200)
        utempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, prior=prior)
        utempo = utempo.item()
        
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} {np.mean(tempo)} {np.mean(utempo)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open('../data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
            
