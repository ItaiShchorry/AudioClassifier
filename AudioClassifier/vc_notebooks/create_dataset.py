# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: audio_cls
#     language: python
#     name: audio_cls
# ---

# + [markdown] Collapsed="false"
# ## Imports

# + Collapsed="false"
import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import tsfresh
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tsfresh import extract_features, feature_selection, extract_relevant_features
# from xgboost import XGBClassifier
import shap
import dask.dataframe as dd
import matplotlib.pyplot as plt
from pathlib import Path

# Load imports

import IPython.display as ipd
import librosa
import librosa.display



# + [markdown] Collapsed="false"
# ## Data

# + Collapsed="false"
AUDIO_FOLDER = Path(r'/dsp/dsp_portal/personal/itai.shchorry/AudioClassificationProject/Udacity-ML-Capstone/UrbanSound Dataset sample/audio')
METADATA_FOLDER = Path(r'/dsp/dsp_portal/personal/itai.shchorry/AudioClassificationProject/Udacity-ML-Capstone/UrbanSound Dataset sample/metadata')

# + Collapsed="false"
import IPython.display as ipd

ipd.Audio(AUDIO_FOLDER/'100032-3-0-0.wav')


# + Collapsed="false"
# load metadata
metadata = pd.read_csv(METADATA_FOLDER/'UrbanSound8K.csv')
metadata.head()

# + Collapsed="false"
# TODO TURN THESE INTO A WIDGET

# Class: Air Conditioner

filename = AUDIO_FOLDER/'100852-0-0-0.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)

# + Collapsed="false"
# Class: Car horn 

filename = AUDIO_FOLDER/'100648-1-0-0.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# + [markdown] Collapsed="false"
# ### Preprocess

# + Collapsed="false"
def extract_mfcc_features(filepath):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return mfccsscaled

# import pyworld as pw
import soundfile as sf

def extract_acoustic_features(filepath):
    data, samplerate = sf.read(filepath, channels=1, endian='LITTLE',dtype='float',subtype='PCM_16',samplerate=48000)
    print("here2, data shape",data.shape,"rate",samplerate)
    f0, sp, ap = pw.wav2world(data, fs=48000)
    f0, sp, ap = min_max_normalize_features(f0, sp, ap)
    f0 = list(map(lambda x: x if x > 0 else 0, f0))
    return f0, sp, ap

def extract_features_with_tsfresh(filepath):
    pass


# + Collapsed="false"
# config
extract_features = extract_mfcc_features

# + Collapsed="false"
metadata.head(1)

# + Collapsed="false"
from pathlib import Path
fulldatasetpath = r'/dsp/dsp_portal/personal/itai.shchorry/AudioClassificationProject/AudioClassifier/Data/UrbanSound8K/audio'
index, row = next(metadata.iterrows())
file_name = Path(fulldatasetpath)/f"fold{row['fold']}"/str(row["slice_file_name"])


# + Collapsed="false"
# Set the path to the full UrbanSound dataset 
fulldatasetpath = r'/dsp/dsp_portal/personal/itai.shchorry/AudioClassificationProject/AudioClassifier/Data/UrbanSound8K/audio'

features = []
idxs = []
# Iterate through each sound file and extract the features 
for index, row in metadata.iterrows():
    
    file_name = Path(fulldatasetpath)/f"fold{row['fold']}"/str(row["slice_file_name"])
    
    class_label = row["class_name"]
    data = extract_features(file_name)
    idx = file_name.parent.name+'/'+file_name.name
    
    features.append([data, class_label])
    idxs.append(idx)

# Convert into a Panda dataframe 
featuresdf = pd.DataFrame(features, columns=['feature','class_label'],index=idxs)

print('Finished feature extraction from ', len(featuresdf), ' files') 

# + Collapsed="false"
from sklearn.preprocessing import LabelEncoder
# from keras.utils import to_categorical

# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
# le = LabelEncoder()
# yy = to_categorical(le.fit_transform(y)) 

# np.save(Path(r'/dsp/dsp_portal/personal/itai.shchorry/AudioClassifier/Data/preprocessed/mfcc/X.npy'),X)
# np.save(Path(r'/dsp/dsp_portal/personal/itai.shchorry/AudioClassifier/Data/preprocessed/mfcc/y.npy'),y)


# + Collapsed="false"
data = pd.DataFrame(X,columns=['mfcc_'+str(i) for i in range(X.shape[1])], index=idxs)
data['y'] = y

# + Collapsed="false"

import pickle
with open(Path(r'/dsp/dsp_portal/personal/itai.shchorry/AudioClassificationProject/AudioClassifier/Data/preprocessed/mfcc/data.p'),'wb') as f:
    pickle.dump(data,f)

