#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 09:09:46 2019
@author: Pedro, Rodrigo, Alix
"""
import numpy as np
import essentia.standard as es
import pywt
import matplotlib.pyplot as plt
# matplotlib inline
import os
import csv

import warnings
warnings.filterwarnings('ignore')

genres = 'classical hiphop jazz reggae rock'.split()

sr = 22050;

g = 'rock'
filename = os.listdir(f'../dataset/{g}')
filename = filename[1]
filename = 'rock.00017.au'
songname = songname = f'../dataset/{g}/{filename}'

data = es.MonoLoader(filename=songname,sampleRate=sr)() # Load song to use ESSENTIA library

#x = np.linspace(0, 1, num=2048)
#chirp_signal = np.sin(250 * np.pi * x**2)
#
#waveletname = 'db4'
#
##data = chirp_signal
#
#band = []
#
#fig, axarr = plt.subplots(nrows=3, ncols=2, figsize=(6,6))
#for ii in range(3):
#    (data, coeff_d) = pywt.dwt(data, waveletname, mode='zero')
#    band.append(coeff_d)
#    if ii==2:
#        band.append(data)
#    axarr[ii, 0].plot(data, 'r')
#    axarr[ii, 1].plot(coeff_d, 'g')
#    axarr[ii, 0].set_ylabel("Level {}".format(ii + 1), fontsize=14, rotation=90)
#    axarr[ii, 0].set_yticklabels([])
#    if ii == 0:
#        axarr[ii, 0].set_title("Approximation coefficients", fontsize=14)
#        axarr[ii, 1].set_title("Detail coefficients", fontsize=14)
#    axarr[ii, 1].set_yticklabels([])
#plt.tight_layout()
#plt.show()
rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
bpm, _, _, _, beats_intervals = rhythm_extractor(data)


BPM_Hist_extractor = es.BpmHistogramDescriptors()
firstPeakBPM, firstPeakWeight, firstPeakSpread, _, _, _, histogram = BPM_Hist_extractor(beats_intervals)

plt.plot(histogram)

#for g in genres:
#    for filename in os.listdir(f'../dataset/{g}'):
#        songname = f'../dataset/{g}/{filename}'
##        audio, _ = librosa.load(songname, mono=True, duration=30)
#        audio = es.MonoLoader(filename=songname,sampleRate=sr)() # Load song to use ESSENTIA library
#        
        