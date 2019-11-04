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
import essentia
import essentia.standard as es
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

## To plot the spectogram of each sample!

#cmap = plt.get_cmap('inferno')
#plt.figure(figsize=(10,10))
#genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
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

#header = 'filename chroma_stft spectral_centroid spectral_bandwidth rolloff zero_crossing_rate tempo'
header  = 'filename bpm danceability pitch_salience'
n_mfcc = 1
for i in range(1, n_mfcc+1):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

file = open('../data_feat.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
    
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

#genres = 'reggae hiphop'.split()

#Using Reggae and HipHop because of higher confusion rate!

#We tried using this feature, but the result is not satisfying 


for g in genres:
    for filename in os.listdir(f'../dataset/{g}'):
        songname = f'../dataset/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        audio = es.MonoLoader(filename=songname)() # Load song for ESSENTIA
#        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
#        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
#        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#       Add TEMPO features
#        onset_env = librosa.onset.onset_strength(y, sr=sr)
#        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
#        tempo = tempo.item()
#        prior = scipy.stats.uniform(40, 200)
#        utempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, prior=prior)
#        utempo = utempo.item()
        
        # Add Features extracted from Essentia
        if(np.size(audio,0)%2!=0):
            audio = audio[:-1]
            
        spectrum_extractor = es.Spectrum()
        spectrum = spectrum_extractor(audio)
        
#        We use BPM to extract rythm features, the main beat of the song
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, _, _, _, beats_intervals = rhythm_extractor(audio)

#        Detrended Fluctuation Analysis of Music Signals
        danceability_extractor = es.Danceability(sampleRate=sr)
        danceability,_ = danceability_extractor(audio)

#        We use pitch salience to check the correlation of the song, i.e if the song is similar to
#        itself in the extract we have. Repeating songs will have higher values of pitch salience
        pitch_salience_extractor = es.PitchSalience(sampleRate=sr)
        pitch_salience = pitch_salience_extractor(spectrum)
        
#        hfc_extract = es.HFC(sampleRate=sr)
#        hfc = hfc_extract(spectrum)
        
#        peak1_bpm, peak1_weight, peak1_spread, peak2_bpm, peak2_weight, peak2_spread, histogram = es.BpmHistogramDescriptors()(beats_intervals)


#        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} {np.mean(tempo)} {np.mean(utempo)}'    
        
        to_append = f'{filename} {np.mean(bpm)} {np.mean(danceability)} {np.mean(pitch_salience)}'  
        
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open('../data_feat.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
            
