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

def download_file_from_bucket(bucket_name, file_path, local_output_path):
    try:
        # Download the file
        response = supabase.storage.from_(bucket_name).download(file_path)
        
        if response:
            # Save the file locally
            with open(local_output_path, "wb") as file:
                file.write(response)
            print(f"File downloaded successfully: {local_output_path}")
        else:
            print(f"Failed to download file: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
bucket_name = "sentiment"
file_path = "vocabulary.txt"  # Path to the file in the bucket
local_output_path = "vocabulary.txt"  # Local file name to save as

download_file_from_bucket(bucket_name, file_path, local_output_path)

# Read the vocabulary file
def load_vocab_from_file(file_path):
    with open(file_path, "r") as file:
        vocab = [line.strip() for line in file.readlines()]  # Read and strip newlines
    return vocab

# File path to the downloaded vocabulary file
vocab_file_path = "vocabulary.txt"

# Load the vocabulary
vocab = load_vocab_from_file(vocab_file_path)

# Set up the TextVectorization layer
MAX_FEATURES = 200000  # Maximum number of tokens
vectorizer = TextVectorization(
    max_tokens=MAX_FEATURES,
    output_sequence_length=1800,  # Sequence length
    output_mode="int"
)

# Assign the vocabulary to the TextVectorization layer
vectorizer.set_vocabulary(vocab)

print("Vocabulary successfully loaded into TextVectorization layer!")

# Define Google Drive file IDs
model_file_id = "1fHukySE-W312ezuiWMfaCDanY6lGWOk8"  # Replace with your model file ID

# Define file paths
model_path = "model.h5"

# Function to download a file from Google Drive
def download_file(file_id, output_path, description):
    if not os.path.exists(output_path):
        with st.spinner(f"Downloading {description}..."):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False)
        st.success(f"{description} downloaded successfully!")
    else:
        st.info(f"{description} already exists.")

# Download model and vocabulary
download_file(model_file_id, model_path, "Model")

# Load the model
model = load_model(model_path)

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
if st.button("Analyze"):
    if user_input.strip():
        if model:
            # Get predictions
            predictions = predict_toxicity(user_input)

            # Display progress bars for each category
            st.subheader("Toxicity Levels:")
            for category, value in zip(categories, predictions):
                st.write(f"{category}: {value * 100:.2f}%")
                st.progress(value)  # Progress bar takes a value between 0 and 1
        else:
            st.error("Model could not be loaded. Please try again.")
    else:
        st.error("Please enter a valid comment.")
