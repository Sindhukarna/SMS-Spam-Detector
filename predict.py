import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load saved model and vectorizer
model = joblib.load(
    os.path.join(BASE_DIR, "models", "spam_model.pkl")
)

vectorizer = joblib.load(
    os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
)

# Get message from user
message = input("Enter an SMS message: ")

# Convert message into numerical features
message_tfidf = vectorizer.transform([message])

# Predict
prediction = model.predict(message_tfidf)

if prediction[0] == 1:
    print("This message is SPAM")
else:
    print("This message is HAM")