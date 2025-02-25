# Comment Toxicity Classification App 💬

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Challenges Solved](#challenges-solved)
- [Future Improvements](#future-improvements)
- [Impact](#impact)
- [Live Demo](#live-demo)


---

## Overview
This deep learning-based application classifies text comments into various toxicity categories—including hate speech, offensive language, and harassment. Built to help moderate online platforms, it filters harmful content in real-time to ensure safer interactions.

---

## Features
- **Multi-category Classification:** Detects toxic, severe toxic, obscene, threat, insult, and identity hate labels.
- **Deep Learning Powered:** Uses an LSTM recurrent neural network for accurate predictions.
- **Interactive Interface:** Developed with Streamlit for a seamless user experience.
- **Real-Time Processing:** Quickly analyzes and classifies user inputs.

---

## Tech Stack
- **Backend:** Python, TensorFlow/PyTorch
- **Frontend:** Streamlit
- **Deployment:** Hosted on Hugging Face Spaces
- **Data Processing:** Leverages pre-trained embeddings for efficient text analysis
- **Storage:** Vocabulary files hosted on Supabase; model available via Google Drive

---

## Installation
1) To get started, clone the repository using `git clone https://github.com/your-repo/comment-toxicity-analyser.git` and navigate into the directory.
2) Next, create and activate a virtual environment (use `python -m venv venv` and activate it using `source venv/bin/activate` for macOS/Linux or `venv\Scripts\activate` for Windows).
3) Then, install the dependencies with `pip install -r requirements.txt`.
4) If you want to download the trained model, here is the gdrive link for the model [https://drive.google.com/file/d/19sR7F2iiyND4bhKaQDYNm5f-cN4c3nq_/view?usp=drive_link]

---

## Usage
After installation, run the application using the command `streamlit run app.py`. Open your browser and navigate to [http://localhost:8501/](http://localhost:8501/) to interact with the app.

---

## Challenges Solved
This project addresses several challenges: it accurately analyzes the context to detect nuanced toxicity, efficiently processes unstructured data including informal language, misspellings, and slang, and maintains balanced performance by reducing false positives and negatives.

---

## Future Improvements
Plans for future enhancements include increasing model accuracy through domain-specific fine-tuning, adding multi-language support, and integrating sentiment analysis for deeper insights.

---

## Impact
By automating toxicity detection, this tool contributes to healthier online interactions and significantly reduces the prevalence of harmful content.

---

## Live Demo
Experience the application in action on Hugging Face Spaces: [**Live Demo**](https://huggingface.co/spaces/ad-2004/comment-toxicity-analyser)


 ![image](https://github.com/user-attachments/assets/e9a9360e-2c0a-4bc7-b7ac-56d0116053cb)

--- 
 
Feel free to contribute, report issues, or suggest improvements!


 

