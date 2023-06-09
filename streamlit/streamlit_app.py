import streamlit as st
import pickle


with open('sentiment-analysis-twitter/sample_data.pickle', 'rb') as f:
    df = pickle.load(f)

print(df.head())