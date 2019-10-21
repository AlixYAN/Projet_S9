# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""


# IMPORT LIBRARIES

# feature extractoring and preprocessing data
import librosa
import librosa.display
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# matplotlib inline
#import os
#from PIL import Image
#import pathlib
#import csv

# Preprocessing
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder, StandardScaler

#Keras
#import keras

import warnings
warnings.filterwarnings('ignore')


# SPECTROGRAM EXTRACTION

# cmap = plt.get_cmap('inferno')

tempo = []

plt.figure(figsize=(10,10))
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
#for g in genres:
    #pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)     
    #for filename in os.listdir(f'./genres/{g}'):
        #songname = f'./genres/{g}/{filename}'
        #y, sr = librosa.load(songname, mono=True, duration=5)
y, sr = librosa.load(librosa.util.example_audio_file())
#        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
#        plt.axis('off');
#        plt.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png')
#        plt.clf()
onset_env = librosa.onset.onset_strength(y, sr=sr)
tempo.append(librosa.beat.tempo(onset_envelope=onset_env, sr=sr))
S, phase = librosa.magphase(librosa.stft(y=y))
cent = librosa.feature.spectral_centroid(y=y, sr=sr)
plt.figure()
plt.subplot(2, 1, 1)
plt.semilogy(cent.T, label='Spectral centroid')
plt.ylabel('Hz')
plt.xticks([])
plt.xlim([0, cent.shape[-1]])
plt.legend()
plt.subplot(2, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time')
plt.title('log Power spectrogram')
plt.tight_layout()
plt.show()
zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        

