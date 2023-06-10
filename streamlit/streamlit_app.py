import streamlit as st
import pickle


with open('./streamlit/sample_data.pickle', 'rb') as f:
    df = pickle.load(f)

st.title('Twitter Sentiment Analysis')
st.text('Sample of dataframe:')
st.dataframe(df.head(10))

with col1:
    st.title('Tweet Prediction')

with col2:
    text_input = st.text_input(
        "Tweet Away 👇",
        placeholder="Enter Tweet Here",
    )

    if text_input:
        st.write("You entered: ", text_input)