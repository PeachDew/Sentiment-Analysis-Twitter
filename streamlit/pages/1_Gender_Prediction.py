import streamlit as st
import pickle
import pandas as pd
import pickle
import webcolors
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
import re
from collections import Counter

st.set_page_config(page_title="Gender Prediction", page_icon="👫")

st.sidebar.header("Predicting your Gender from viewing your profile.")
st.markdown("""
# Gender/Brand Prediction 📱
This app utilizes machine learning to make predictions based on a Twitter user profile. Simply provide us with some information about a Twitter user, and we'll generate a prediction for you!
"""
)

# Load the models
with open("./Gender_Model_Save/best_clf.pkl", "rb") as file:
    name_model = pickle.load(file)
with open("./Gender_Model_Save/logregtxt.pkl", "rb") as file:
    text_model = pickle.load(file)
with open("./Gender_Model_Save/logregdesc.pkl", "rb") as file:
    desc_model = pickle.load(file)
with open("./Gender_Model_Save/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

col_names = ['name_pred','red_ratio','green_ratio',
             'blue_ratio','fav_number','tweet_count',
             'desc_pred','pred_pred','uppercase_count']

        

def hex_to_rgb(hex_code):
    # hex_code = '#' + hex_code 
    try:
        rgb = webcolors.hex_to_rgb(hex_code)
    except ValueError:
        hex_code = "#0084B4"
        rgb = webcolors.hex_to_rgb(hex_code)
    sum_rgb = sum(rgb)
    ratios = [comp / sum_rgb for comp in rgb]
    return ratios

values = [None for i in range(len(col_names))]
col1, col2 = st.columns(2)
with col1:
    colc, cold = st.columns([3,1])
    with colc:
        name = st.text_input('Twitter Username', '',
                              placeholder = 'Twitter Username')
        uppercase_count = sum(1 for char in name if char.isupper())
        values[8] = uppercase_count
        name_processed = re.sub('[^a-zA-Z0-9]', '', name).lower()
        boc = vectorizer.transform([name_processed])
        name_pred = name_model.predict(boc.toarray())[0]
        values[0] = name_pred
        
        
    with cold:
        color = st.color_picker('Twitter Link Color', '#1DA1F2')
        values[1], values[2], values[3] = hex_to_rgb(color)
        st.write(values)
    
    cola, colb = st.columns(2)
    with cola:
        favno = st.number_input('Favorite number', value=42)
    with colb:
        tweets = st.number_input('Number of tweets', value=500)

    desc = st.text_area('Twitter description', placeholder='I love farming!')

    txt = st.text_area('Paste a random tweet from your account:',
                       placeholder='Feelin good at the sunny beach B)')
    
with col2:
    pred_button = st.button('Generate Prediction')
    if pred_button:
        if title and favno and tweets and desc and txt:
            st.balloons()
            st.write("Prediction here")
        else:
            st.error("Please fill in all the input fields.")

        
st.markdown("### LDA (Latent Dirichlet Allocation)")
st.write("LDA (Latent Dirichlet Allocation) is an algorithm used for topic modeling, a method that helps uncover the hidden themes and patterns within a collection of documents. It's like having a detective investigating a library full of books, trying to figure out the different topics covered. LDA assumes that each document is a mixture of various topics, and each topic is characterized by a distribution of words. By carefully examining the words and their frequencies, LDA helps us identify and understand the underlying themes present in the documents. It's a powerful tool for organizing and making sense of large volumes of text data.")
output_file = "./visualisations/lda_visualization.html"
st.components.v1.html(open(output_file, 'r', encoding='utf-8').read(), height=700, width=800, scrolling=True)


