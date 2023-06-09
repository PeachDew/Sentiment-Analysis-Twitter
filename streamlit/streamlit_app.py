import streamlit as st
import pickle


with open('./streamlit/sample_data.pickle', 'rb') as f:
    df = pickle.load(f)

st.title('Twitter Sentiment Analysis')
st.table(df)