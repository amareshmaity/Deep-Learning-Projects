import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
from keras.models import load_model

import streamlit as st

## Load the word index
max_features=10000
word_index = imdb.get_word_index()
reversed_word_index = {value: key for key, value in word_index.items()}

## Load the pre-trained model with tanh activation 
model = load_model("./artifacts/model.h5")

## Helper function (utility functions)
import re

def decode_review(encoded_review):
    decoded_tokens = []
    for token in encoded_review:
        if token == 0:
            continue
        decoded_tokens.append(reversed_word_index.get(token - 3, '?'))
    return " ".join(decoded_tokens)

def preprocess_text(text):
    words = re.findall(r"\b\w+\b", text.lower())
    encoded_review = []
    for word in words:
        token = word_index.get(word)
        if token is None:
            encoded_review.append(2)  # unknown token
        elif token + 3 < max_features:
            encoded_review.append(token + 3)
        else:
            encoded_review.append(2)  # outside vocabulary limit
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

## Creating prediction function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment, prediction[0][0]

## Streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative.")

## User input
user_input = st.text_area("Movie Review")

if st.button('Classify'):
    sentiment, prediction = predict_sentiment(user_input)

    ## Display result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction}')
else:
    st.write('Please enter a movie review.')

