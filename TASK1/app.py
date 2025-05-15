import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download necessary resources
nltk.download('punkt')

def extractive_summarization(text, num_sentences=3):
    sentences = sent_tokenize(text)
    
    # Edge case: text has fewer sentences than requested
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    sentence_vectors = vectorizer.fit_transform(sentences)

    # Cosine similarity matrix
    similarity_matrix = cosine_similarity(sentence_vectors)

    # Sentence scores (sum of similarities to other sentences)
    sentence_scores = similarity_matrix.sum(axis=1)

    # Get indices of top N sentence scores
    top_sentence_indices = np.argsort(sentence_scores)[-num_sentences:]

    # Sort the selected indices to maintain original order
    top_sentence_indices = sorted(top_sentence_indices)

    # Form the final summary
    summary = " ".join([sentences[i] for i in top_sentence_indices])
    return summary

# Streamlit App
st.title("Extractive Text Summarization")
user_input = st.text_area("Enter the text to summarize:")
num_sentences = st.slider("Number of sentences in summary:", 1, 10, 3)

if st.button("Summarize"):
    if user_input.strip():
        summary = extractive_summarization(user_input, num_sentences)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter text to summarize.")
