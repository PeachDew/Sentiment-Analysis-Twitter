import streamlit as st
import numpy as np
import pickle
import pandas as pd
import pickle
import webcolors
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import re
from collections import Counter

import gensim
from gensim.models import Word2Vec

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

nltk.download('stopwords')

st.set_page_config(page_title="Gender Prediction", page_icon="👫")

st.sidebar.header("Predicting Gender from Twitter Profile.")
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
with open("./Gender_Model_Save/desc_w2v_model.pkl", "rb") as file:
    desc_w2v_model = pickle.load(file)
with open("./Gender_Model_Save/text_w2v_model.pkl", "rb") as file:
    text_w2v_model = pickle.load(file)    
with open("./Gender_Model_Save/final_model.pkl", "rb") as file:
    final_model = pickle.load(file)    
with open("./Gender_Model_Save/final_df.pkl", "rb") as file:
    final_df = pickle.load(file)     
with open("./Gender_Model_Save/final_y.pkl", "rb") as file:
    final_y = pickle.load(file)       
    
final_df['gender'] = final_y    

col_names = ['name_pred','red_ratio','green_ratio',
             'blue_ratio','fav_number','tweet_count',
             'desc_pred','text_pred','uppercase_count']

        

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

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

values = [None for i in range(len(col_names))]
col1, col2 = st.columns(2)
with col1:
    colc, cold = st.columns([3,1])
    with colc:
        name = st.text_input('Twitter Username', '',
                              placeholder = 'Twitter Username')
    with cold:
        color = st.color_picker('Link Color', '#1DA1F2')
    
    cola, colb = st.columns(2)
    with cola:
        favno = st.number_input('Favorite number', value=42)
    with colb:
        tweets = st.number_input('Number of tweets', value=500)                   
    desc = st.text_area('Twitter description', placeholder='I love farming!')    

    txt = st.text_area('Paste a random tweet from your account:',
                       placeholder='Feelin good at the sunny beach B)')
    
with col2:
    colb1, colb2, colb3 = st.columns([1,4,1])
    with colb2:
        pred_button = st.button('Generate Prediction')
    if pred_button:
        with st.spinner('Wait for it...'):
            uppercase_count = sum(1 for char in name if char.isupper())
            values[8] = uppercase_count
            name_processed = re.sub('[^a-zA-Z0-9]', '', name).lower()
            boc = vectorizer.transform([name_processed])
            name_pred = name_model.predict(boc.toarray())[0]
            values[0] = int(name_pred)

            values[1], values[2], values[3] = hex_to_rgb(color)
            values[4] = favno
            values[5] = tweets

            default_embedding = np.zeros(desc_w2v_model.vector_size)

            dtokens = gensim.utils.simple_preprocess(desc)
            dlemmatized_tokens = [lemmatizer.lemmatize(token) for token in dtokens]
            filtered_tokens_desc = [token for token in dlemmatized_tokens if token not in stop_words]
            dembeddings = [desc_w2v_model.wv[token] if token in desc_w2v_model.wv else default_embedding
                           for token in filtered_tokens_desc]
            if len(dembeddings) > 0:
                mean_desc_embed = np.mean(dembeddings, axis=0)
            else: 
                mean_desc_embed = default_embedding

            desc_pred = desc_model.predict([mean_desc_embed])[0]
            values[6] = desc_pred

            ttokens = gensim.utils.simple_preprocess(txt)
            tlemmatized_tokens = [lemmatizer.lemmatize(token) for token in ttokens]
            filtered_tokens_txt = [token for token in tlemmatized_tokens if token not in stop_words]
            tembeddings = [text_w2v_model.wv[token] if token in text_w2v_model.wv else default_embedding
                           for token in filtered_tokens_txt]
            if len(tembeddings) > 0:
                mean_text_embed = np.mean(tembeddings, axis=0)
            else: 
                mean_text_embed = default_embedding

            text_pred = text_model.predict([mean_text_embed])[0]
            values[7] = text_pred


            if name and favno and tweets and desc and txt:
                data = {col_name: [value] for col_name, value in zip(col_names, values)}
                desired_order = ['fav_number', 'text_pred', 'desc_pred',
                                 'name_pred', 'red_ratio', 'green_ratio',
                                 'blue_ratio', 'tweet_count', 'uppercase_count']
                df = pd.DataFrame(data)
                df = df.reindex(columns=desired_order)
                gender_pred = final_model.predict_proba(df)[0]

                st.balloons()
                if (gender_pred[0] > gender_pred[1]) and (gender_pred[0] > gender_pred[2]):
                    col5, col6 = st.columns([3,2])
                    with col5:
                        st.markdown("## <div style='color: #F7A9FF; text-align: center;'>Girl 👧</div>", unsafe_allow_html=True)
                    with col6:
                        st.markdown(f"## <div style='color: #F7A9FF; text-align: center; '>{gender_pred[0]*100:.3g}%</div>", unsafe_allow_html=True)
                    col3, col4 = st.columns(2)
                    with col3:
                         st.markdown(f"### <div style='color: #FFEBCC; text-align: center; '>Brand 🏢 {gender_pred[2]*100:.3g}%</div>", unsafe_allow_html=True)
                    with col4:
                        st.markdown(f"### <div style='color: #7EA7FF; text-align: center;'>Boy 👦 {gender_pred[1]*100:.3g}%</div>", unsafe_allow_html=True)


                elif (gender_pred[1] > gender_pred[0]) and (gender_pred[1] > gender_pred[2]):
                    col5, col6 = st.columns([3,2])
                    with col5:
                        st.markdown("## <div style='color: #7EA7FF; text-align: center;'>Boy 👦</div>", unsafe_allow_html=True)
                    with col6:
                        st.markdown(f"## <div style='color: #7EA7FF; text-align: center; '>{gender_pred[1]*100:.3g}%</div>", unsafe_allow_html=True)
                    col3, col4 = st.columns(2)
                    with col3:
                        st.markdown(f"### <div style='color: #F7A9FF; text-align: center;'>Girl 👧 {gender_pred[0]*100:.3g}%</div>", unsafe_allow_html=True)
                    with col4:
                        st.markdown(f"### <div style='color: #FFEBCC; text-align: center; '>Brand 🏢 {gender_pred[2]*100:.3g}%</div>", unsafe_allow_html=True)


                else:
                    col5, col6 = st.columns([3,2])
                    with col5:
                        st.markdown("## <div style='color: #FFEBCC; text-align: center; '>Brand🏢</div>", unsafe_allow_html=True)
                    with col6:
                        st.markdown(f"## <div style='color: #FFEBCC; text-align: center; '>{gender_pred[2]*100:.3g}%</div>", unsafe_allow_html=True)
                    col3, col4 = st.columns(2)
                    with col3:
                        st.markdown(f"### <div style='color: #F7A9FF; text-align: center;'>Girl 👧 {gender_pred[0]*100:.3g}%</div>", unsafe_allow_html=True)
                    with col4:
                        st.markdown(f"### <div style='color: #7EA7FF; text-align: center;'>Boy 👦 {gender_pred[1]*100:.3g}%</div>", unsafe_allow_html=True)

            else:
                st.error("Please fill in all the input fields.")


st.markdown("## How it works 😋")
st.markdown('''
### Text Preprocessing 🪚
Here are various techniques I used to prepare the text data for further analysis. This involved several key steps:

- Tokenization: Dividing text into individual tokens or words.
''')
st.code(''' "I ate a book!" --> "I" "ate" "a" "book!" ''')
st.markdown('''            
- Lowercasing: Converting all text to lowercase to ensure consistency and reduce the impact of case sensitivity.
''')
st.code('''"I" "ate" "a" "book!" --> "i" "ate" "a" "book!"''')
st.markdown('''     
- Lemmatization: Reducing words to their base or dictionary form, such as converting 'running' to 'run'.
''')
st.code('''"i" "ate" "a" "book!" --> "i" "eat" "a" "book!"''')
st.markdown('''     
Removing Stop Words: Eliminating common stop words from the text, such as 'the', 'and', and 'is', which do not contribute significant meaning to the analysis. List of words included in nltk's stopwords can be found [here](https://gist.github.com/sebleier/554280).
''')
st.code('''"i" "eat" "a" "book!" --> "eat" "book!"''')
st.markdown('''     
- Removing Special Characters: Eliminate non-alphanumeric characters, symbols, and punctuation marks to focus on the essential textual content.
''')
st.code('''"eat" "book!" --> "eat" "book"''')
st.markdown('''     
These steps helped to clean and standardize the text data.
''')
st.markdown("## The Data: Columns, Classes 😋")
st.dataframe(final_df.head(10))


#st.markdown("### LDA (Latent Dirichlet Allocation)")
#st.write("LDA (Latent Dirichlet Allocation) is an algorithm used for topic modeling, a method that helps uncover the hidden themes and patterns within a collection of documents. It's like having a detective investigating a library full of books, trying to figure out the different topics covered. LDA assumes that each document is a mixture of various topics, and each topic is characterized by a distribution of words. By carefully examining the words and their frequencies, LDA helps us identify and understand the underlying themes present in the documents.)
#output_file = "./visualisations/lda_visualization.html"
#st.components.v1.html(open(output_file, 'r', encoding='utf-8').read(), height=700, width=800, scrolling=True)


