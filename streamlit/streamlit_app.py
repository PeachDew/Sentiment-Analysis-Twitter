import streamlit as st
import pickle
import os

absolute_path = os.getcwd()
data_pickle = 'sample_data.pickle'
file_path = absolute_path + '/' + data_pickle

with open(file_path, 'rb') as f:
    df = pickle.load(f)

print(df.head())