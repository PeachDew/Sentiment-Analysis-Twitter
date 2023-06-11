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
    color_discrete_map = {}
    for i in range(1, 32):
        darkness = i / 31  
        r = int(210 * darkness)  
        g = int(151 * darkness)  
        b = int(252 * darkness)  
        color = f"#{r:02x}{g:02x}{b:02x}" 
        color_discrete_map[i] = color
    fig = px.histogram(df, x="day", color="day", color_discrete_map=color_discrete_map)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
with tab3:
    fig = px.histogram(df, x="month", color = 'month', 
                       color_discrete_map = {4: '#c367e6',
                                             5: '#b75cd9',
                                             6: '#ab51cd'})
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)