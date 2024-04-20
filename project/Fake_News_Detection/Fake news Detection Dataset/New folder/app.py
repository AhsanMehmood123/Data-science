import streamlit as st
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords # the for of in with remove these words not useful for our project 
from nltk.stem.porter import PorterStemmer # loved, loving == love 
from sklearn.feature_extraction.text import TfidfVectorizer # convert word into (love = [0.0])
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # using for classification problem
from sklearn.metrics import accuracy_score

# Load data
new_df = pd.read_csv('train.csv')
new_df = new_df.fillna(' ') # to fill the null values with space
new_df['content'] = new_df['author']+" "+new_df['title'] # to join author and title colum in one colum content
x = new_df.drop('label', axis=1)
y = new_df['label']

# Define stemming function
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Apply stemming function to content column
new_df['content'] = new_df['content'].apply(stemming)

# Vectorize data
x = new_df['content'].values
y = new_df['label'].values
vector = TfidfVectorizer()
vector.fit(x)
x= vector.transform(x)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)

# Fit logistic regression model
model = LogisticRegression()
model.fit(x_train,y_train)

# website
st.title('Fake News Detector')
input_text = st.text_input('Enter news Article')

def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.write('The News is Fake')
    else:
        st.write('The News Is Real')