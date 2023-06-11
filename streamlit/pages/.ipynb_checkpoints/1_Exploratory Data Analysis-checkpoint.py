import streamlit as st
import pickle

st.set_page_config(page_title="EDA", page_icon="📈")

st.sidebar.header("Exploring the Data")
st.write(
    """Here we go through some data visualisation, cleaning, and feature engineering techniques, 
    enjoy!"""
)

with open('./streamlit/sample_data.pickle', 'rb') as f:
    df = pickle.load(f)

st.markdown('## Sample of dataframe:')
st.dataframe(df.head(10))     