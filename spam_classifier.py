# spam_classifier.py (CV-ready version)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib  # for saving/loading

# 1. Load CSV (ignore extra columns)
df = pd.read_csv('data/spam.csv', encoding='latin-1', usecols=[0, 1], names=['label', 'message'], header=0)

# 2. Preprocess text
df['message'] = df['message'].str.lower()  # lowercase
df['message'] = df['message'].str.replace(r'[^\w\s]', '', regex=True)  # remove punctuation

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# 4. Vectorize text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6. Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Save model and vectorizer
joblib.dump(model, 'models/spam_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
print("Model and vectorizer saved to 'models/' folder.")

# 8. Optional: Load model and test new messages
model = joblib.load('models/spam_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

while True:
    msg = input("Enter a message to classify (or 'exit' to quit): ").lower()
    if msg == 'exit':
        break
    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)
    print("Prediction:", prediction[0])