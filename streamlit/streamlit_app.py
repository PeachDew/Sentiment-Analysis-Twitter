import streamlit as st
import pickle
import os

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

filename = file_selector()
st.write('You selected `%s`' % filename)

with open(filename, 'rb') as f:
    df = pickle.load(f)

print(df.head())