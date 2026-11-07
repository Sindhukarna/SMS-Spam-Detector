\# SMS Spam Detector



\## Project Overview

SMS Spam Detector is a machine learning web application that classifies SMS messages as Spam or Ham (normal messages).



Users can enter an SMS message and receive a prediction with a probability score.



\## Technologies Used

\- Python

\- Pandas

\- Scikit-learn

\- TF-IDF Vectorization

\- Multinomial Naive Bayes

\- Joblib

\- Streamlit



\## Machine Learning Workflow

1\. Load and explore the SMS dataset.

2\. Encode labels (Ham = 0, Spam = 1).

3\. Split data into training and testing sets.

4\. Convert text messages into numerical features using TF-IDF.

5\. Train a Multinomial Naive Bayes model.

6\. Evaluate the model using accuracy, confusion matrix and classification report.

7\. Save the trained model and TF-IDF vectorizer using Joblib.



\## Model Performance

\- Accuracy: 96.05%

\- Algorithm: Multinomial Naive Bayes



\## Features

\- SMS Spam/Ham prediction

\- Spam/Ham probability display

\- Prediction progress bar

\- Clear Message button

\- Prediction History

\- Clear History button

\- Streamlit web interface



\## How to Run



Install the required libraries:



```bash

pip install pandas scikit-learn joblib streamlit

```



Run the application:



```bash

python -m streamlit run app.py

```



\## Project Structure



```text

spam/

│

├── datasets/

│   └── SMSSpamCollection

│

├── models/

│   ├── spam\_model.pkl

│   └── tfidf\_vectorizer.pkl

│

├── first\_step.py

├── predict.py

├── app.py

└── README.md

```



\## Conclusion

This project demonstrates how machine learning and natural language processing can be used to identify unwanted SMS messages.

