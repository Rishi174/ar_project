import streamlit as st
import pandas as pd
import numpy as np
import librosa
import os
import pickle
import keras
from pydub import AudioSegment


mapping = {0:'cough_deep', 1:'cough_shallow', 2:'breath_deep', 3:'breath_shallow'}

def generate_features(data, sample_rate):
    mfcc_features = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
    mfcc_scaled_features = np.mean(mfcc_features.T,axis=0)
    return mfcc_scaled_features

def export_to_wav_file(file, name):
    f = AudioSegment.from_wav(file)
    f.export(f'{name}.wav', format = 'wav')


def process_data(files):
    feat_array = np.array([])
    for i, file in enumerate(files):
        export_to_wav_file(file, mapping[i])
        data,sample_rate=librosa.load(f'{mapping[i]}.wav')
        feat_array = np.concatenate((feat_array,generate_features(data, sample_rate)), axis = 0)
    return feat_array
    

nn_model = keras.models.load_model('aug_basic_model_2022_10_2_20_47_17')
#nn_model = keras.models.load_model('../model_weights/aug_basic_model_2022_10_2_20_47_17')

st.title('Covid Classification')

st.write('Please provide metadata information')

cough_heavy = st.file_uploader('upload heavy cough file')
cough_shallow = st.file_uploader('upload shallow cough file')
breath_deep = st.file_uploader('upload deep breathing file')
breath_shallow = st.file_uploader('upload shallow breathing file')
all_files = [cough_heavy, cough_shallow, breath_deep, breath_shallow]

if cough_heavy:
    if cough_shallow:
        if breath_deep:
            if breath_shallow:
                feats_data = (process_data(all_files)).reshape(1,-1)
                prob = nn_model.predict(feats_data)
                if prob[0][0] > 0.25:
                    st.write('The following recordings belong to covid patient')
                else:
                    st.write('The following recordings belong to non covid patient')
                
                
                for i in mapping.keys():
                    os.remove(f'{mapping[i]}.wav')
    
    
