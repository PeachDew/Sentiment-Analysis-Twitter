import streamlit as st
import pickle
import pandas as pd


st.set_page_config(page_title="Gender Prediction", page_icon="👫")

st.sidebar.header("Predicting your Gender from viewing your profile.")
st.write(
    """Description here."""
)

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
    
with col2:
    pred_button = st.button('Generate Prediction')
    if pred_button:
        st.write("Prediction here")

        
st.markdown("###LDA (Latent Dirichlet Allocation)")
st.write("LDA (Latent Dirichlet Allocation) is an algorithm used for topic modeling, a method that helps uncover the hidden themes and patterns within a collection of documents. It's like having a detective investigating a library full of books, trying to figure out the different topics covered. LDA assumes that each document is a mixture of various topics, and each topic is characterized by a distribution of words. By carefully examining the words and their frequencies, LDA helps us identify and understand the underlying themes present in the documents. It's a powerful tool for organizing and making sense of large volumes of text data.")
output_file = "./visualisations/lda_visualization.html"
st.components.v1.html(open(output_file, 'r', encoding='utf-8').read(), height=500, width = 1000)


