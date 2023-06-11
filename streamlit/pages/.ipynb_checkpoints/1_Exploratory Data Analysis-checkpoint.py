import streamlit as st
import pickle

st.set_page_config(page_title="EDA", page_icon="📈")

st.sidebar.header("Exploring the Data")
st.write(
    """Here we go through some data visualisation, cleaning, and feature engineering techniques, 
    enjoy!"""
)

st.markdown('## Sample of dataframe:')
st.dataframe(df.head(10))     