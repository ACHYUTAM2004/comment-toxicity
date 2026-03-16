# 💬 Comment Toxicity Classification App

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow?style=for-the-badge)](https://huggingface.co/spaces/ad-2004/comment-toxicity-analyser)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?style=for-the-badge\&logo=python)](https://www.python.org/)

This deep learning-based application classifies text comments into various toxicity categories—including hate speech, offensive language, and harassment. Built to help moderate online platforms, it filters harmful content in real-time to ensure safer interactions.

---

## 📸 Demo

Here's a quick look at the application in action:

![App Screenshot](https://github.com/user-attachments/assets/e9a9360e-2c0a-4bc7-b7ac-56d0116053cb)

---

## ✨ Features

* **Multi-Label Classification:** Detects `toxic`, `severe toxic`, `obscene`, `threat`, `insult`, and `identity hate` labels.
* **Deep Learning Powered:** Uses an LSTM recurrent neural network for accurate contextual predictions.
* **Interactive Interface:** Developed with Streamlit for a seamless and responsive user experience.
* **Real-Time Processing:** Quickly analyzes and classifies user inputs on the fly.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend & AI:** Python, TensorFlow/Keras
* **Data Processing:** Pre-trained text embeddings
* **Storage:** Vocabulary files hosted on Supabase; model available via Google Drive

---

## 🚀 Getting Started

Follow these steps to get the project running on your local machine.

1. **Clone the Repository:**

```bash
git clone https://github.com/your-repo/comment-toxicity-analyser.git
cd comment-toxicity-analyser
```

2. **Create and Activate a Virtual Environment:**

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

3. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

4. **Download the Pre-trained Model:**

The model is hosted on Google Drive. Please download the `.h5` model file from this link and place it in the root directory of the project.

https://drive.google.com/file/d/19sR7F2iiyND4bhKaQDYNm5f-cN4c3nq_/view?usp=drive_link

5. **Run the Application:**

```bash
streamlit run app.py
```

---

## 📝 How to Use

After installation, run the application using `streamlit run app.py` and navigate to `http://localhost:8501`. Enter any comment or text into the input box and click **"Analyze"** to see the toxicity classification.

---

## 🧠 Model Architecture

The toxicity classifier is built using a **Recurrent Neural Network (RNN)** architecture with **Long Short-Term Memory (LSTM)** layers to capture contextual relationships in text sequences.

### Model Overview

The model processes raw text comments and converts them into numerical sequences using a **TextVectorization layer**. These sequences are then fed into an LSTM-based neural network that learns patterns associated with toxic language.

The system performs **multi-label classification**, meaning a single comment can belong to multiple toxicity categories simultaneously.

### Architecture Components

**1. Text Preprocessing**

* Input comments are converted into token sequences using **Keras TextVectorization**.
* Vocabulary is dynamically loaded from **Supabase storage**.
* Each comment is padded/truncated to a fixed sequence length.

**2. Embedding Layer**

* Words are converted into dense vector representations.
* Captures semantic relationships between words.

**3. LSTM Layer**

* Learns contextual dependencies between words in a sequence.
* Particularly effective for detecting nuanced toxicity patterns.

**4. Dense Output Layer**

* Fully connected layer with **sigmoid activation**.
* Produces probability scores for each toxicity category.

### Output Labels

The model predicts probabilities for the following categories:

* **Toxic**
* **Severe Toxic**
* **Obscene**
* **Threat**
* **Insult**
* **Identity Hate**

Each label receives a probability score between **0 and 1**, representing the likelihood of that toxicity type being present in the comment.

### Prediction Pipeline

1. User enters a comment in the Streamlit interface.
2. Text is vectorized using the preloaded vocabulary.
3. The trained LSTM model processes the sequence.
4. Probability scores are generated for all toxicity categories.
5. Results are displayed with percentages and progress bars.

The Streamlit interface in the project code handles user input, preprocessing, model inference, and result visualization in real time. 

---

## 🎯 Project Context

### Challenges Solved

This project accurately analyzes the context to detect nuanced toxicity, efficiently processes unstructured data including informal language and slang, and maintains balanced performance by reducing false positives and negatives.

### Future Improvements

Plans for future enhancements include increasing model accuracy through domain-specific fine-tuning, adding multi-language support, and integrating sentiment analysis for deeper insights.

### Impact

By automating toxicity detection, this tool contributes to healthier online interactions and significantly reduces the manual moderation workload on online platforms.

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or find a bug, please feel free to open an issue or submit a pull request.
