import streamlit as st
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


st.set_page_config(page_title="EDA", page_icon="📈")

st.sidebar.header("Exploring the Data")
st.write(
    """Here we go through some data visualisation, cleaning, and feature engineering techniques, 
    enjoy!"""
)

with open('./streamlit/sample_data.pickle', 'rb') as f:
    df = pickle.load(f)

st.markdown('## Sample of dataframe:')
st.markdown("Positive tweets' target value is 4, while negative tweets' is 0") 
st.dataframe(df.head(10))    

st.markdown('## Frequency of tweets by Hour, Day, and Month')
st.markdown("All data retrieved from 2009, from months April to June.")

tab1, tab2, tab3 = st.tabs(["Hour", "Day", "Month"])

with tab1:
    base_color = "#b9fa93"
    num_values = 24

    color_discrete_map = {}
    for i in range(0, num_values + 1):
        darkness = 0.6 + i / (num_values+35)  
        r, g, b = tuple(int(base_color[i:i + 2], 16) for i in (1, 3, 5))  
        r = int(r * darkness)  
        g = int(g * darkness) 
        b = int(b * darkness)  
        color = f"#{r:02x}{g:02x}{b:02x}"  
        color_discrete_map[i] = color
        
    fig = px.histogram(df, x="hour", color="hour", color_discrete_map=color_discrete_map)
    fig.update(layout_showlegend=False)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
with tab2:
    base_color = "#97a4fc"
    num_values = 31 

    color_discrete_map = {}
    for i in range(1, num_values + 1):
        darkness = 0.5 + i / (num_values+31)  
        r, g, b = tuple(int(base_color[i:i + 2], 16) for i in (1, 3, 5))  
        r = int(r * darkness)  
        g = int(g * darkness) 
        b = int(b * darkness)  
        color = f"#{r:02x}{g:02x}{b:02x}"  
        color_discrete_map[i] = color
        
    fig = px.histogram(df, x="day", color="day", color_discrete_map=color_discrete_map)
    fig.update(layout_showlegend=False)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
with tab3:
    fig = px.histogram(df, x="month", color = 'month', 
                       color_discrete_map = {4: '#c367e6',
                                             5: '#b75cd9',
                                             6: '#ab51cd'})
    fig.update(layout_showlegend=False)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)


st.markdown('## Separating positive and negative tweets:')    
positive_tweets = df[df['target'] == 4]
negative_tweets = df[df['target'] == 0]    
tab1, tab2, tab3 = st.tabs(["Hour", "Day", "Month"])

with tab1:
    positive_tweets_by_hour = positive_tweets.groupby('hour').size().reset_index(name='positive_count')
    negative_tweets_by_hour = negative_tweets.groupby('hour').size().reset_index(name='negative_count')

    merged_counts = positive_tweets_by_hour.merge(negative_tweets_by_hour, on='hour', how='outer').fillna(0)

    fig = go.Figure(data=[
        go.Bar(name='Positive', x=merged_counts['hour'], y=merged_counts['positive_count']),
        go.Bar(name='Negative', x=merged_counts['hour'], y=merged_counts['negative_count'])
    ])

    fig.update_layout(xaxis={'title': 'Hour'}, yaxis={'title': 'Count'},
                      title='Number of Positive and Negative Tweets by Hour',
                      barmode='group')

    st.plotly_chart(fig, use_container_width=True)
    
with tab2:
    positive_tweets_by_day = positive_tweets.groupby('day').size().reset_index(name='positive_count')
    negative_tweets_by_day = negative_tweets.groupby('day').size().reset_index(name='negative_count')

    merged_counts = positive_tweets_by_day.merge(negative_tweets_by_day, on='day', how='outer').fillna(0)

    fig = go.Figure(data=[
        go.Bar(name='Positive', x=merged_counts['day'], y=merged_counts['positive_count']),
        go.Bar(name='Negative', x=merged_counts['day'], y=merged_counts['negative_count'])
    ])

    fig.update_layout(xaxis={'title': 'Day'}, yaxis={'title': 'Count'},
                      title='Number of Positive and Negative Tweets by Day of Month',
                      barmode='group')

    st.plotly_chart(fig, use_container_width=True)

with tab3:
    positive_tweets_by_month = positive_tweets.groupby('month').size().reset_index(name='positive_count')
    negative_tweets_by_month = negative_tweets.groupby('month').size().reset_index(name='negative_count')

    merged_counts = positive_tweets_by_month.merge(negative_tweets_by_month, on='month', how='outer').fillna(0)

    fig = go.Figure(data=[
        go.Bar(name='Positive', x=merged_counts['month'], y=merged_counts['positive_count']),
        go.Bar(name='Negative', x=merged_counts['month'], y=merged_counts['negative_count'])
    ])

    fig.update_layout(xaxis={'title': 'Month'}, yaxis={'title': 'Count'},
                      title='Number of Positive and Negative Tweets by Month',
                      barmode='group')

    st.plotly_chart(fig, use_container_width=True)
    
st.markdown('''Based on the analysis of the Twitter data, positive tweets tend to peak around midnight (in PDT) while negative tweets show a peak in the early morning hours of 6-9am (in PDT). Additionally, there is a noticeable pattern of increased positivity at the start and end of each month, while sentiments become more negative during the period of the 15th to the 25th of each month. 

It is worth noting that the data exhibits gaps on certain specific dates, which raises questions about the data collection process. Further investigation is needed to ensure the quality and reliability of the data. 

Notably, April and May show a higher proportion of positive tweets, while June stands out with a higher number of negative tweets.
''')