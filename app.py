import streamlit as st
import numpy as np
import tensorflow as tf  # or import your ML framework
from tensorflow.keras.layers import TextVectorization
import gdown
import os
from tensorflow.keras.models import load_model  # adjust for your framework

# Define Google Drive file ID and model path
file_id = "your_file_id_here"  # Replace with your file ID
output_path = "model.h5"  # The filename for the downloaded model

# Function to download the model from Google Drive
def download_model(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(output_path):
        with st.spinner("Downloading model... This may take a while."):
            gdown.download(url, output_path, quiet=False)
    else:
        st.info("Model already downloaded.")

# Load the model
def load_toxicity_model(path):
    if os.path.exists(path):
        return load_model(path)
    else:
        st.error("Model file not found.")
        return None

# Streamlit App
st.title("Comment Toxicity Classifier")

# Step 1: Download the model
download_model(file_id, output_path)

# Step 2: Load the model
model = load_toxicity_model(output_path)

# Load the pre-trained model
model = load_model('path_to_your_model.h5')  # Replace with your model path

# Define the categories
categories = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]

def predict_toxicity(comment):
    # Preprocess the comment (tokenization, padding, etc., as per your model's requirements)
    processed_comment = preprocess_comment(comment)  # Implement this based on your training
    prediction = model.predict(processed_comment)
    return prediction[0]  # Assuming prediction is a 2D array [[...]]; get the first element

def preprocess_comment(comment):
    MAX_FEATURES = 200000
    vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')
    return vectorizer([comment])
# Streamlit App Interface
st.title("Comment Toxicity Classifier")
st.write("Enter a comment to analyze its toxicity levels across six categories.")

# Input text box
user_input = st.text_area("Enter your comment:")

if st.button("Analyze"):
    if user_input.strip():
        # Get predictions
        predictions = predict_toxicity(user_input)

        # Display progress bars for each category
        st.subheader("Toxicity Levels:")
        for category, value in zip(categories, predictions):
            st.write(f"{category}: {value * 100:.2f}%")
            st.progress(value)  # Progress bar takes a value between 0 and 1
    else:
        st.error("Please enter a valid comment.")
