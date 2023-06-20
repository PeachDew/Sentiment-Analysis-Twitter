import streamlit as st
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


st.set_page_config(page_title="Gender Prediction", page_icon="👨👩")

st.sidebar.header("Predicting your Gender from viewing your profile.")
st.write(
    """Description here."""
)

with open('./streamlit/sample_data.pickle', 'rb') as f:
    df = pickle.load(f)
