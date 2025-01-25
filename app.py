import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
import os
import gdown
import requests

def download_vocab_from_github(url, output_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, "w") as file:
            file.write(response.text)
        print(f"Vocabulary file downloaded successfully to {output_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

# URL of the vocab.txt file on GitHub (raw format)
vocab_url = "https://raw.githubusercontent.com/ACHYUTAM2004/comment-toxicity/refs/heads/main/vocabulary.txt"
local_output_path = "vocabulary.txt"

download_vocab_from_github(vocab_url, local_output_path)

# Load the vocabulary after downloading
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

# Function to download the model from Google Drive
def download_model():
    url = "https://drive.google.com/uc?export=download&id=1fHukySE-W312ezuiWMfaCDanY6lGWOk8"  # Replace with your model's Google Drive URL
    output = "Trained_model.h5"  # Local path where the model will be saved
    gdown.download(url, output, quiet=False)

# Function to load the model after downloading it
def load_model_from_drive():
    download_model()  # Download the model from Google Drive
    model = tf.keras.models.load_model("Trained_model.h5")  # Load the model
    return model

# Example usage:
model = load_model_from_drive()

# Now you can use the `model` for predictions
print("Model loaded successfully!")

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
