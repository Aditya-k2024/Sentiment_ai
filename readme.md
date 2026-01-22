#  Sentiment AI 

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Operational-brightgreen?style=for-the-badge)

A high-performance **Natural Language Processing (NLP)** web application that analyzes movie reviews to detect sentiment (Positive/Negative). 

It features a **Matrix style** interface, real-time confidence visualization, and a robust machine learning backend trained on 50,000 IMDB reviews.

### Streamlit live app : https://sentimentai123.streamlit.app/

---

## ⚡ Features

* **Matrix-Style UI:** A futuristic, "hacker-style" interface built with Streamlit and custom CSS.
* **Advanced Preprocessing:** Full NLP pipeline including HTML cleaning, POS-Tagging, and Lemmatization.
* **High Accuracy Model:** Powered by **Linear Regression** achieving **~88.61% accuracy**, also tried LinearSVM with **~88.21%** and Naive Bayes with **~84%**  on the IMDB dataset.
* **Interactive Visualization:** Dynamic Plotly donut charts that change color (Green/Red) based on sentiment confidence.
* **Real-time Analysis:** Instant classification of any text input.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit, Plotly (for charts), HTML/CSS injection.
* **Machine Learning:** Scikit-Learn (LinearSVC, Logistic Regression).
* **Natural Language Processing:** NLTK (WordNet Lemmatizer, POS Tagging).
* **Data Handling:** Pandas, Numpy, Joblib.

---

## 📂 Project Structure

```text
📂 Sentiment-AI
│
├── 📂 Notebook & models          
│   ├── 📄 IMDB Dataset.csv       
│   ├── 📄 lr_model.pkl           
│   ├── 📄 movie_review.ipynb     
│   └── 📄 tf_vectorizer.pkl      
│
├── 📄 app.py                     
├── 📄 requirements.txt           
└── 📄 readme.md                  
```

## Installation & Setup

### 1. Clone the Repository
```text
git clone [https://github.com/Aditya-k2024/Sentiment_ai)
cd movie-sentiment-ai
```

### 2. Install Dependencies
```text
pip install -r requirements.txt
```

### 3. Download NLTK Data (First Run Only)
```text
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
``` 

### 4. Run the Application
```text
streamlit run app.py
```

---

## ❓ How It Works (The Pipeline) 

* **Input**: The user enters a raw text review (e.g., "The visual effects were stunning, but the story was weak.").


* **Cleaning**:

    HTML tags are removed.

    Non-alphabetic characters are stripped.

    Text is converted to lowercase.


* **NLP Processing**:

    Tokenization: Splits text into words.

    POS Tagging: Identifies parts of speech (Noun, Verb, Adjective).

    Lemmatization: Converts words to their root form using WordNet (e.g., "running" -> "run").


* **Vectorization**: The cleaned text is transformed into numbers using the TF-IDF Vectorizer loaded from Notebook & models/tf_vectorizer.pkl.


* **Prediction**: The model loaded from Notebook & models/lr_model.pkl predicts the class (0 or 1) and calculates a confidence score.


* **Output**: The dashboard updates with the result and a visual confidence gauge.


---
