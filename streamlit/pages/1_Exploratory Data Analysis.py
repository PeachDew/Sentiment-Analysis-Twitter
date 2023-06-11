import streamlit as st
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


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

st.markdown('## Distribution of date elements:')
st.markdown("All data retrieved from 2009, from months April to June.")

tab1, tab2, tab3 = st.tabs(["Hour", "Day", "Month"])

with tab1:
    fig = px.histogram(df, x="hour")
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
with tab2:
    fig = px.histogram(df, x="day")
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
with tab3:
    fig = px.histogram(df, x="month", color_discrete_map = {0: '#f394ff',
                                                            1: '#e889ff',
                                                            2: '#dc7eff',
                                                            3: '#d172f2',
                                                            4: '#c367e6',
                                                            5: '#b75cd9',
                                                            6: '#ab51cd',
                                                            7: '#9f46c1',
                                                            8: '#933bb4',
                                                            9: '#872fad',
                                                            10: '#7b24a1',
                                                            11: '#6f198f',
                                                            12: '#63147c'})
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)