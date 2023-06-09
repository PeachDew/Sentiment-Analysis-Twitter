import streamlit as st
import pickle

with open('sample_data.pickle', 'rb') as f:
    df = pickle.load(f)

df.head()