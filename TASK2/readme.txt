# Chatbot Analytics Dashboard

This project focuses on building a basic analytics dashboard for chatbot interactions using a custom-trained NLP model. The app is built with **Streamlit** and helps track metrics such as the number of user queries, common topics, and user feedback.

## 🔍 Features

- ✅ Extract intent from user queries using a trained ML model
- ✅ Track number of queries submitted
- ✅ Identify most common topics (intents)
- ✅ Analyze user satisfaction using positive/negative feedback
- ✅ Streamlit GUI for easy interaction

## 📁 Project Structure

├── app.py                # Streamlit GUI app
├── train_model.ipynb     # Jupyter Notebook for model training and evaluation
├── model.pkl             # Trained classification model
├── vectorizer.pkl        # TF-IDF vectorizer used for feature extraction
├── label_encoder.pkl     # LabelEncoder for mapping intents to numbers
├── chatbot_intents.csv   # Training dataset with query-intent pairs
├── requirements.txt      # List of required Python packages
└── README.md             # Project documentation


## 📊 Model Info

- Algorithm Used: Logistic Regression (or similar)
- Vectorizer: TF-IDF
- Accuracy: ≥ 70%

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt

###2. Run the Streamlit App
streamlit run app.py

📌 Dependencies
Python 3.8+
scikit-learn
pandas
streamlit

📈 Example Intents
weather_query — “What is the weather like today?”

book_flight — “Book a flight to New York”

positive_feedback — “Great service!”

negative_feedback — “This isn't helpful”

## 💬 Example Queries to Try

Below are some example inputs that you can give to the chatbot:

- "Book me a flight to Mumbai"
- "Cancel my last order"
- "What’s the weather tomorrow?"
- "Play some jazz music"
- "Schedule a meeting for Monday"
- "Where is my refund?"
- "Turn on the lights"

These inputs will be processed by the trained model, and a predicted **intent** will be displayed, along with options to rate the response.
