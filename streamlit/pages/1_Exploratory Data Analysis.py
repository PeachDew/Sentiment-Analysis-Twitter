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

months = df.month
days = df.day
hours = df.hour

st.markdown('## Distribution of date elements:')
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
sns.histplot(days, kde=False, ax=axes[0],binwidth=0.5)
axes[0].set_title('Distribution by Day')
axes[0].set_xlabel('Day')
axes[0].set_ylabel('Count')
axes[0].set_xticks(range(1, 31, 4))
sns.histplot(months, kde=False, ax=axes[1],binwidth=0.5)
axes[1].set_title('Distribution by Month')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Count')
axes[1].set_xticks(range(1, 12))
sns.histplot(hours, kde=False, ax=axes[2],binwidth=0.5)
axes[2].set_title('Distribution by Hour')
axes[2].set_xlabel('Hour')
axes[2].set_ylabel('Count')
axes[2].set_xticks(range(0,23,4))
st.pyplot(fig)

fig = px.histogram(df, x="day")
st.plotly_chart(fig, theme="streamlit", use_container_width=True)