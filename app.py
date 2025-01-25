import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
import os
import gdown
from supabase import create_client, Client
import os

# Replace these with your Supabase project details
SUPABASE_URL = "https://vbgxuijebobixzrqgvys.supabase.co"  # Your Supabase project URL
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZiZ3h1aWplYm9iaXh6cnFndnlzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzU4ODk1MjIsImV4cCI6MjA1MTQ2NTUyMn0.xchbHvyHL3Y1EQ5SQbKMA--CtVlRXsPNUieXTSRZYPY"  # Your Supabase service role key (or API key)

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Download vocabulary file from bucket
def download_file_from_bucket(bucket_name, file_path, local_output_path):
    try:
        response = supabase.storage.from_(bucket_name).download(file_path)
        if response:
            with open(local_output_path, "wb") as file:
                file.write(response)
            print(f"File downloaded successfully: {local_output_path}")
        else:
            print(f"Failed to download file: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example: Download vocabulary
bucket_name = "sentiment"
file_path = "vocabulary/vocabulary.txt"
local_output_path = "vocabulary.txt"
download_file_from_bucket(bucket_name, file_path, local_output_path)

# Load vocabulary
def load_vocab_from_file(file_path):
    with open(file_path, "r") as file:
        vocab = [line.strip() for line in file.readlines()]
    return vocab

# Initialize TextVectorization globally
try:
    vocab_file_path = "vocabulary.txt"
    vocab = load_vocab_from_file(vocab_file_path)

    # Configure TextVectorization layer
    MAX_FEATURES = 200000
    vectorizer = TextVectorization(
        max_tokens=MAX_FEATURES,
        output_sequence_length=1800,
        output_mode="int"
    )
    vectorizer.set_vocabulary(vocab)  # Assign vocabulary
    st.info("Vocabulary successfully loaded into TextVectorization layer!")
except Exception as e:
    st.error(f"Failed to initialize vectorizer: {e}")
    vectorizer = None  # Fallback if initialization fails

# Define Google Drive model file download
model_file_id = "1fHukySE-W312ezuiWMfaCDanY6lGWOk8"
model_path = "model.h5"

def download_file(file_id, output_path, description):
    if not os.path.exists(output_path):
        with st.spinner(f"Downloading {description}..."):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False)
        st.success(f"{description} downloaded successfully!")
    else:
        st.info(f"{description} already exists.")

# Download model
download_file(model_file_id, model_path, "Model")
model = load_model(model_path)

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
