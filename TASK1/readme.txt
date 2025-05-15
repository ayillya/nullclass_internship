# Extractive Text Summarization App - NullClass Internship Task 1

## Overview
This project implements an extractive text summarization system using TF-IDF and cosine similarity. It extracts the most relevant sentences from the input text to create a concise summary.

The project includes:
- A Jupyter notebook (`summarizer_model_training.ipynb`) for training the TF-IDF vectorizer model.
- A Streamlit app (`app.py`) as the graphical user interface (GUI) for users to input text and get summaries.
- Saved model file (`saved_model/vectorizer.pkl`) containing the trained TF-IDF vectorizer.
- A `requirements.txt` file listing the necessary Python dependencies.

## Folder Structure
Task1/
│
├── app.py
├── summarizer_model_training.ipynb
├── saved_model/
│ └── vectorizer.pkl
├── requirements.txt
└── README.md

## How to Run

### 1. Clone the repository
git clone <your-github-repo-url>
cd Task1

### 2. Install dependencies
pip install -r requirements.txt

###3. Run the Streamlit app
streamlit run app.py
This will open the summarization app in your default web browser.

About the Model:
*The model uses TF-IDF vectorization of sentences and cosine similarity to score and rank sentences by importance.

*The summary length can be controlled via a slider in the app.

*The TF-IDF vectorizer is trained on the input text dynamically, no external pretrained models are used.

Notes:
*This is an extractive summarization technique and works best with relatively well-structured input text.

*Unlike classification tasks, standard accuracy metrics (precision, recall, confusion matrix) do not apply here.

*Evaluation can be done qualitatively or via text summarization metrics like ROUGE (not implemented here).