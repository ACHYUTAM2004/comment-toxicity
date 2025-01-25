import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
import os
import gdown
import requests


def download_model():
    url = "https://drive.google.com/file/d/1fHukySE-W312ezuiWMfaCDanY6lGWOk8/view?usp=sharing"
    output = "Trained_model.h5"
    gdown.download(url, output, quiet=False)
    
# Load the model after downloading
def load_model():
    download_model()
    model = tf.keras.models.load_model("Trained_model.h5")
    return model

# Preprocess the input text
def preprocess_comment(comment):
    return vectorizer([comment])  # Transform the text input into vectorized form

# Prediction function
def predict_toxicity(comment):
    processed_comment = preprocess_comment(comment)
    prediction = model.predict(processed_comment)
    return prediction[0]

# Streamlit App
st.title("Comment Toxicity Classifier")

# User input
user_input = st.text_area("Enter a comment to check for toxicity:", "")
if user_input:
    predictions = predict_toxicity(user_input)

    # Display predictions with progress bars
    categories = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]
    st.subheader("Toxicity Levels:")
    for category, score in zip(categories, predictions):
        st.text(f"{category}: {score * 100:.2f}%")
        st.progress(score)
