import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
import os
import gdown
from supabase import create_client, Client

# Supabase Configuration
SUPABASE_URL = "https://vbgxuijebobixzrqgvys.supabase.co"  # Replace with your Supabase URL
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZiZ3h1aWplYm9iaXh6cnFndnlzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzU4ODk1MjIsImV4cCI6MjA1MTQ2NTUyMn0.xchbHvyHL3Y1EQ5SQbKMA--CtVlRXsPNUieXTSRZYPY"  # Replace with your Supabase Service Role Key
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Function to download the model from Google Drive
def download_model():
    model_path = "Trained_model.h5"
    if not os.path.exists(model_path):  # Check if the model file already exists
        url = "https://drive.google.com/uc?export=download&id=1fHukySE-W312ezuiWMfaCDanY6lGWOk8"
        gdown.download(url, model_path, quiet=False)
    else:
        st.info("Model already exists locally.")
    return model_path

# Function to fetch the vocabulary from Supabase
def fetch_vocab():
    response = supabase.storage.from_("sentiment").download("vocabulary/vocab.txt")
    if response:
        vocab_content = response.decode("utf-8")  # Decode the byte content to a string
        vocab_list = vocab_content.splitlines()  # Convert to a list of words
        return vocab_list
    else:
        st.error("Failed to fetch vocabulary from Supabase.")
        return []

# Function to initialize the TextVectorization layer
def initialize_vectorizer(vocab):
    vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=200, vocabulary=vocab)
    return vectorizer

# Load the model and vocabulary
model_path = download_model()
model = load_model(model_path)
vocab = fetch_vocab()
vectorizer = initialize_vectorizer(vocab) if vocab else None

# Preprocess function
def preprocess_comment(comment):
    if vectorizer:
        return vectorizer([comment])  # Use vectorizer globally
    else:
        st.error("Vectorizer is not initialized. Cannot process comments.")
        return None

# Prediction function
def predict_toxicity(comment):
    processed_comment = preprocess_comment(comment)
    if processed_comment is not None:
        prediction = model.predict(processed_comment)
        return prediction[0]
    else:
        return []

# Streamlit app
st.title("Comment Toxicity Classifier")
categories = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]

user_input = st.text_area("Enter a comment to check for toxicity:")
if st.button("Analyze"):
    if user_input.strip():
        if model:
            predictions = predict_toxicity(user_input)
            if predictions:
                st.subheader("Toxicity Levels:")
                for category, value in zip(categories, predictions):
                    st.write(f"{category}: {value * 100:.2f}%")
                    st.progress(value)
            else:
                st.error("Prediction failed. Please check the input.")
        else:
            st.error("Model could not be loaded.")
    else:
        st.error("Please enter a valid comment.")
