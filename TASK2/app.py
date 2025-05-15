import streamlit as st
import pandas as pd
import pickle
import joblib
from collections import Counter
import matplotlib.pyplot as plt

# Load trained model, vectorizer, and label encoder
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

le = joblib.load("label_encoder.pkl")

# In-memory session storage
if 'queries' not in st.session_state:
    st.session_state.queries = []
if 'intents' not in st.session_state:
    st.session_state.intents = []
if 'ratings' not in st.session_state:
    st.session_state.ratings = []

st.title("🤖 Chatbot Analytics Dashboard")
st.markdown("""
### 📝 Example Queries You Can Try:
- *"What is the weather like today?"*
- *"Book a flight to Delhi"*
- *Give me today's headlines*
- *"Play some music"*
- *"I love using this chatbot!"*
""")

# --- Input Query ---
query = st.text_input("Enter a user query:")

if query:
    X_input = vectorizer.transform([query])
    predicted_code = model.predict(X_input)[0]
    predicted_label = le.inverse_transform([predicted_code])[0]

    st.success(f"Predicted Intent: **{predicted_label}**")

    st.session_state.queries.append(query)
    st.session_state.intents.append(predicted_label)

    rating = st.selectbox("How satisfied are you with the response?", ["Select", "Satisfied", "Neutral", "Unsatisfied"])
    if rating != "Select":
        st.session_state.ratings.append(rating)
        st.info("Feedback submitted!")

st.divider()

# --- Analytics Section ---
st.header("📊 Analytics Summary")

st.write(f"**Total Queries:** {len(st.session_state.queries)}")

# Most common topics
if st.session_state.intents:
    top_intents = Counter(st.session_state.intents).most_common()
    df_intents = pd.DataFrame(top_intents, columns=["Intent", "Count"])
    st.subheader("Most Common Predicted Topics")
    st.bar_chart(df_intents.set_index("Intent"))

# Satisfaction Breakdown
if st.session_state.ratings:
    st.subheader("User Satisfaction Ratings")
    rating_counts = Counter(st.session_state.ratings)
    df_ratings = pd.DataFrame(rating_counts.items(), columns=["Rating", "Count"])
    st.bar_chart(df_ratings.set_index("Rating"))
