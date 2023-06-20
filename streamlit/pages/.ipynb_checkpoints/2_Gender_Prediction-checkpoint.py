import streamlit as st
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


st.set_page_config(page_title="Gender Prediction", page_icon="👫")

st.sidebar.header("Predicting your Gender from viewing your profile.")
st.write(
    """Description here."""
)

with open('./streamlit/sample_data.pickle', 'rb') as f:
    df = pickle.load(f)

col1, col2 = st.columns(2)
with col1:
    title = st.text_input('Twitter Username', 'Lydia23')
    st.write('Your username is:', title)

    favno = st.number_input('Favorite number', value=42)
    st.write('Your favorite number is:', favno)

    tweets = st.number_input('Number of tweets', value=500)
    st.write('You tweeted this many times:', tweets)

    desc = st.text_area('Twitter description', 'I love farming!')
    st.write("Your description:", desc)

    txt = st.text_area('Paste a random tweet from your account:', 'Feelin good at the sunny beach B)')
    st.write("Your tweet:", txt)

    color = st.color_picker('Your twitter link color', '#1DA1F2')
    st.write('The current color is', color)
    
    pred_button = st.button('Generate')
    
with col2:
    if pred_button:
        st.write("Prediction here")


