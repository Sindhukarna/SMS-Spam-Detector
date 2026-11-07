
import pandas as pd
import os
from sklearn.model_selection import train_test_split


# Get the current project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset path
file_path = os.path.join(BASE_DIR, "datasets", "SMSSpamCollection")

# Check whether dataset exists
if not os.path.exists(file_path):
    print("Dataset not found!")
    print("Please check the datasets folder.")
else:
    # Load dataset
    data = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["label", "message"]
    )

    # Display dataset information
    print("Dataset loaded successfully!")
    print("\nFirst 5 rows:")
    print(data.head())

    print("\nDataset shape:", data.shape)

    print("\nMissing values:")
    print(data.isnull().sum())
    
    # Convert labels into numbers
    data["label"] = data["label"].map({"ham":0,"spam":1})

    # Display updated labels
    print("\nAfter Label Encoding:")
    print(data.head())

    print("\nLabel counts:")
    print(data["label"].value_counts())

# Separate messages and labels
X = data["message"]
y = data["label"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display dataset sizes
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)

#convert message to numerical values
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


from sklearn.naive_bayes import MultinomialNB

# Create Naive Bayes model
model = MultinomialNB()

# Train the model
model.fit(X_train_tfidf, y_train)

print("Naive Bayes model trained successfully!")
# Make predictions
y_pred = model.predict(X_test_tfidf)

print("Predictions:")
print(y_pred)

#ckecking accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100)

#how many are correctly classified
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

from sklearn.metrics import classification_report

print("Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Ham", "Spam"]
))

# Test our own SMS messages

new_messages = [
    "Congratulations! You won a free prize. Claim now!",
    "Hey, are you coming to college today?",
    "URGENT! You have won a cash reward. Call now!"
]

# Convert messages into TF-IDF features
new_messages_tfidf = vectorizer.transform(new_messages)

# Predict the results
new_predictions = model.predict(new_messages_tfidf)

# Display results
for message, prediction in zip(new_messages, new_predictions):

    if prediction == 1:
        print("Spam:", message)
    else:
        print("Ham:", message)

import joblib
import os

# Create paths for saving
model_path = os.path.join(BASE_DIR, "models", "spam_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

# Save model and vectorizer
joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print("Model saved successfully!")
print("Vectorizer saved successfully!")