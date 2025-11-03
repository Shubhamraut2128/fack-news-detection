import streamlit as st
import pickle
import re

# -----------------------------
# Load model and vectorizer
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)          # remove URLs
    text = re.sub(r"[^A-Za-z\s]", "", text)      # remove punctuation/numbers
    text = text.lower().strip()                  # lowercase
    return text

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection App")
st.write("Enter any news article text below to check whether it's **Real** or **Fake**.")

user_input = st.text_area("Enter news text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned_text = clean_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text]).toarray()
        prediction = model.predict(vectorized_text)[0]

        if prediction == 0:
            st.error("❌ This news is **FAKE**.")
        else:
            st.success("✅ This news is **REAL**.")
