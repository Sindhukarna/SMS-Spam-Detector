import streamlit as st
import joblib
import os

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "models", "spam_model.pkl")
)

vectorizer = joblib.load(
    os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
)

if "history" not in st.session_state:
    st.session_state.history = []

def clear_message():
    st.session_state.sms_input = ""

st.title("📩 SMS Spam Detector")
st.subheader("Protect yourself from unwanted messages")
st.info("Enter an SMS message below to check whether it is Spam or Ham.")


message = st.text_area(
    "Enter your SMS message:",
    key="sms_input"
)

col1, col2 = st.columns(2)

with col1:
    check_button = st.button(
        "Check Message",
        use_container_width=True
    )

with col2:
    st.button(
        "Clear Message",
        on_click=clear_message,
        use_container_width=True
    )
if check_button:

    if message.strip():

        message_tfidf = vectorizer.transform([message])

        prediction = model.predict(message_tfidf)
        result = "SPAM" if prediction[0] == 1 else "HAM"

        st.session_state.history.append({
            "Message": message,
            "Result": result
        })

        probabilities = model.predict_proba(message_tfidf)[0]

        spam_probability = probabilities[1] * 100
        ham_probability = probabilities[0] * 100

        if prediction[0] == 1:
            st.error("SPAM MESSAGE DETECTED")
            st.write("Be careful! This message may be unwanted or fraudulent.")
            st.write(f"Spam probability: {spam_probability:.2f}%")
            st.progress(spam_probability / 100)

        else:
            st.success("SAFE MESSAGE - HAM")
            st.write("This message is classified as a normal SMS.")
            st.write(f"Ham probability: {ham_probability:.2f}%")
            st.progress(ham_probability / 100)
    else:
        st.warning("Please enter a message.")


st.subheader("Prediction History")

if st.session_state.history:
    st.table(st.session_state.history)
else:
    st.write("No messages checked yet.")

if st.button("Clear History"):
    st.session_state.history = []
    st.rerun()