# spam_app.py

import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load('models/spam_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

st.title("SMS Spam Classifier")
st.write("Type a message below and see if it is spam or ham.")

# Input box
user_input = st.text_area("Enter your message:")

if st.button("Classify"):
    if user_input.strip() != "":
        msg_vec = vectorizer.transform([user_input])
        prediction = model.predict(msg_vec)[0]
        st.success(f"Prediction: **{prediction}**")
    else:
        st.warning("Please enter a message to classify.")