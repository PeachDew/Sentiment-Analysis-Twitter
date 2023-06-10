import streamlit as st
import pickle
import sklearn


with open('./streamlit/sample_data.pickle', 'rb') as f:
    df = pickle.load(f)
    
with open('./streamlit/tfidf_logreg_model.pickle', 'rb') as f:
    best_logreg_model = pickle.load(f)
    
with open('./streamlit/vectorizer.pickle', 'rb') as f:
    vectorizer = pickle.load(f)

st.title('Twitter Sentiment Analysis')
st.text('Sample of dataframe:')
st.dataframe(df.head(10))

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    # tokenize text
    tokens = word_tokenize(text)
    # removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
    # lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

def predict_tweet_probability(tweet, model):
    cleaned_tweet = preprocess_text(tweet)
    print(cleaned_tweet)
    cleaned_tweet = ' '.join(cleaned_tweet)
    print(cleaned_tweet)
    transformed_tweet = vectorizer.transform([cleaned_tweet])
    # Predict the probability of the tweet being positive
    probability = model.predict_proba(transformed_tweet)[:, 1]
    return probability[0]

col1, col2 = st.columns(2)

with col1:
    st.title('Tweet Prediction')

with col2:
    text_input = st.text_input(
        "Tweet Away 👇",
        placeholder="Enter Tweet Here",
    )

    if text_input:
        probability = predict_tweet_probability(tweet, best_logreg_model)

        if probability > 0.5:
            print(f"Probability that the tweet is positive: {probability * 100:.3g}%")
        else:
            print(f"Probability that the tweet is negative: {(1 - probability) * 100:.3g}%")