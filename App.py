import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model

# Load model
model = load_model('simple_rnn_model.h5')

# Parameters
vocab_size = 10000
maxlen = 100

# Load word index
word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = {v: k for k, v in word_index.items()}

# Function to convert review text to sequence of word indices
def encode_review(text):
    tokens = text.lower().split()
    encoded = [1]  # start token
    for word in tokens:
        index = word_index.get(word, 2)  # 2 is <UNK>
        if index < vocab_size:
            encoded.append(index)
    return pad_sequences([encoded], maxlen=maxlen, padding='pre')

# Streamlit UI
st.title("🎬 IMDB Movie Review Sentiment Classifier")
st.write("Enter a short movie review to classify it as positive or negative using an RNN model.")

# Input box
user_input = st.text_area("Enter your review here:", height=150)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        encoded_input = encode_review(user_input)
        prediction = model.predict(encoded_input)[0][0]
        sentiment = "👍 Positive" if prediction >= 0.5 else "👎 Negative"
        st.subheader("Prediction:")
        st.write(f"**{sentiment}** (Confidence: {prediction:.2f})")
