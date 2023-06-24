import streamlit as st
import numpy as np
import pickle
import pandas as pd
import pickle
import webcolors
import re
import matplotlib.pyplot as plt

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
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

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
with open("./Gender_Model_Save/full_df.pkl", "rb") as file:
    full_df = pickle.load(file)          
    
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
st.markdown("## The Data: Columns, and Classes 📊")
final_df = final_df[['gender'] + list(final_df.columns[:-1])]
colns = ['gender','fav_number','text','description','name','red_ratio', 'green_ratio','blue_ratio','tweet_count','uppercase_count']
st.markdown('''
We have the target variable in the first column, 'gender' with 3 possible values: 0 for female, 1 for male, and 2 for brand.''')
st.dataframe(full_df[colns].head(10),hide_index=True)

st.markdown("### Dealing with Twitter Link color 🎨")
st.markdown('''
I first convert the color in hexadecimal to its corresponding RGB values. Then, to incorporate some form of interaction between the three terms, I used the ratio of a color among all 3 RGB colors. Below we have a simple example of the process.''')
color1, color2 = st.columns(2)
with color1:
    color_demo = st.color_picker('Link Color', '#1DA1F2', key = 42)
with color2:
    crr1, crr2 = st.columns(2)
    with crr1:
        col_button = st.button('Get Ratios')
    with crr2:
        if col_button:
            r, g, b = hex_to_rgb(color)
            y = np.array([r,g,b])
            mycolors = ["red", "green", "blue"]
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.grid(False)
            plt.pie(y, colors=mycolors)
            fig.set_facecolor("none")
            plt.axis("equal")  # Ensure the pie chart is circular
            plt.savefig("plot.png", dpi=300, transparent=True, bbox_inches="tight")
            st.image("plot.png", use_column_width=True)
    if col_button:
        with st.spinner('Wait for it...'):
            if color_demo:
                r, g, b = hex_to_rgb(color)
                
                data = [{'Red ratio': r, 'Green ratio': g, 'Blue ratio': b}]
                col_df = pd.DataFrame(data)
                st.dataframe(col_df.head(), hide_index=True)   
            else:
                st.error("Please fill in all the input fields.")
st.markdown('''
### Numerical Features:
### Favorite number, Tweet count, and Uppercase count #️⃣
I considered using one-hot encoding for favorite number, as numbers can represent more than their numerical value. For example 7 being known as a lucky number, or 42 being the meaning of life, or 13 being an unlucky number. However, I decided against it as it would increase the dimensionality substantially, and there might still be patterns to be learnt between the magnitude of the number and the target. 
Uppercase count, and tweet count can be inputs to a ML model as they are.''')
st.image("https://i.pinimg.com/originals/ba/86/d3/ba86d3e30b6562392434eda0baaa1cc4.jpg", caption="https://themindsjournal.com/repeating-numbers-and-their-meanings/")

st.markdown('''### Text Features: Sample Tweet, Bio, and Username 🔤
To leverage the semantic meaning of the text fields, Word2Vec was employed to create word/character embeddings. Three separate models were trained, each dedicated to one of the categories, allowing for the capture of unique patterns and characteristics within the text data. These models were then utilized to predict the target variable on the training data, generating individual predictions for each category.
Here is a video that helped me understand how Word2Vec works:
''')
st.video("https://www.youtube.com/watch?v=viZrOnJclY0")

st.markdown("### Putting it all together")
st.markdown("Here is how the final dataframe looks like:")
st.dataframe(final_df.head(7), hide_index=True)
st.markdown('''As you can see, we have 3 prediction columns, and 3 ratio columns for the colors. Finally, I tuned a catboost model using Random Search using the simple code shown below. To cover more iterations I decided against K-fold CV and used a simple validation set. "cat_features" indicates the parameters to be treated as categorical, and loss_function='MultiClass' is set for classification tasks like this one.''')
st.code('''
param_grid = {
    'learning_rate': np.linspace(0.01, 0.1, 40),
    'depth': list(range(6, 10)),
    'l2_leaf_reg': list(range(2, 20)),
    'iterations': [500],
    'early_stopping_rounds': [20],
    'thread_count': [-1]
}

X_train, X_val, y_train, y_val = train_test_split(final_df_train, final_train_y, test_size=0.1, random_state=42)

best_score = 0
top_models = []

num_iterations = 50
tqdm._instances.clear()
progress_bar = tqdm(total=num_iterations, desc='RandomizedSearchCV')

for _ in range(num_iterations):
    # randomly sample hyperparameters
    params = {param: np.random.choice(values) for param, values in param_grid.items()}
    
    catboost_model = CatBoostClassifier(cat_features=[1,2,3],
                                        loss_function='MultiClass', 
                                        **params)
    catboost_model.fit(X_train, y_train, verbose=False)
    
    # evaluate
    val_predictions = catboost_model.predict(X_val)
    score = accuracy_score(y_val, val_predictions)
    
    # update
    if score > best_score:
        best_score = score
        best_model = catboost_model
    
    # store the model and parameters
    top_models.append((catboost_model, params, score))
    
    progress_bar.update(1)

progress_bar.close()

# Sort the models based on the scores
top_models.sort(key=lambda x: x[2], reverse=True)

print("Best Model:", best_model)
print("Best Score:", best_score)
print("Best Hyperparameters:", best_model.get_params())''', language='python')
st.markdown("Click the github symbol at the top right of the page to view the full code available on my github repo! 😊")


#st.markdown("### LDA (Latent Dirichlet Allocation)")
#st.write("LDA (Latent Dirichlet Allocation) is an algorithm used for topic modeling, a method that helps uncover the hidden themes and patterns within a collection of documents. It's like having a detective investigating a library full of books, trying to figure out the different topics covered. LDA assumes that each document is a mixture of various topics, and each topic is characterized by a distribution of words. By carefully examining the words and their frequencies, LDA helps us identify and understand the underlying themes present in the documents.)
#output_file = "./visualisations/lda_visualization.html"
#st.components.v1.html(open(output_file, 'r', encoding='utf-8').read(), height=700, width=800, scrolling=True)


