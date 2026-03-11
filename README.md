# SMS Spam Classifier

A simple machine learning project that classifies SMS messages as **spam** or **ham (not spam)** using Python. The project includes preprocessing, training, evaluation, and an interactive web demo built with Streamlit.
 <img width="837" height="481" alt="image" src="https://github.com/user-attachments/assets/bf577868-3393-44d4-8c93-6b8b4e235534" />

---

## Features
- Text preprocessing: lowercasing and punctuation removal
- Train/test split and evaluation of model performance
- Naive Bayes classifier achieving ~98% accuracy
- Saved model and vectorizer for quick predictions
- Interactive Streamlit web app for classifying new messages

---

## Installation & Usage

1. **Clone the repository**
```bash
git clone https://github.com/CodeByAzriel/sms-spam-classifier.git
cd sms-spam-classifier

Install dependencies

pip install pandas scikit-learn streamlit joblib

Run the Streamlit app

streamlit run spam_app.py

Use the interactive input

Type any message and click Classify

The app will show whether the message is spam or ham
