# app.py
import streamlit as st
import pickle
import nltk
nltk.download('stopwords')

# Load model
with open('model/model.pkl', 'rb') as f:
    model, vectorizer = pickle.load(f)

# UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.markdown("Enter a news article below to check if it's **FAKE** or **REAL**.")

user_input = st.text_area("📝 News Article", height=200)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        news_vec = vectorizer.transform([user_input])
        prediction = model.predict(news_vec)[0]
        label = "REAL" if prediction == 1 else "FAKE"
        st.success(f"🧠 Prediction: **{label}**")
        proba = model.predict_proba(news_vec)[0]
        confidence = max(proba)
        st.info(f"🔍 Confidence: **{confidence:.2f}**")
        st.write(f"📊 FAKE: {proba[0]:.2f}, REAL: {proba[1]:.2f}")

# Show both class probabilities
st.write(f"Raw prediction: {prediction}")
