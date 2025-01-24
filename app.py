import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
import os
import gdown

# Define Google Drive file IDs
model_file_id = "1fHukySE-W312ezuiWMfaCDanY6lGWOk8"  # Replace with your model file ID
vocab_file_id = "1utE7JF-ZUaP3wRFV0HL_e1_rhsE5Oe-0"  # Replace with your vocabulary file ID

# Define file paths
model_path = "model.h5"
vocab_path = "vocabulary.txt"

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
download_file(vocab_file_id, vocab_path, "Vocabulary")

# Load the model
model = load_model(model_path)

# Initialize the vectorizer
MAX_FEATURES = 200000
vectorizer = TextVectorization(
    max_tokens=MAX_FEATURES,
    output_sequence_length=1800,
    output_mode="int"
)

# Load vocabulary into the vectorizer
if os.path.exists(vocab_path):
    with open(vocab_path, "r") as f:
        vocab = [line.strip() for line in f.readlines()]
    vectorizer.set_vocabulary(vocab)
else:
    st.error("Vocabulary file not found.")

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
