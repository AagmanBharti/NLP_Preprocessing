import streamlit as st
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ---------------- LOAD SPACY ----------------
nlp = spacy.load("en_core_web_sm")

# ---------------- REGEX PREPROCESSING ----------------
def regex_preprocess(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove emails
    text = re.sub(r'\S+@\S+', ' ', text)

    # Remove emojis and special characters
    text = re.sub(r'[^\w\s]', ' ', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', ' ', text)

    # Extract alphabetic tokens
    tokens = re.findall(r'\b[a-z]+\b', text)

    return tokens

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="NLP Preprocessing", layout="wide")
st.title("NLP Text Preprocessing Application")
st.write("Regex-based Cleaning, Tokenization, Stemming, Lemmatization, BoW, TF-IDF")

# ---------------- USER INPUT ----------------
text = st.text_area(
    "Enter Text for NLP Processing",
    height=150,
    placeholder="Example: Email me at abc@gmail.com 😊 Visit https://nlp.ai"
)

# ---------------- SIDEBAR ----------------
option = st.sidebar.radio(
    "Select NLP Technique",
    [
        "Tokenization",
        "Text Cleaning (Regex)",
        "Stemming",
        "Lemmatization",
        "Bag of Words (Regex)",
        "TF-IDF (Regex)"
    ]
)

# ---------------- PROCESS BUTTON ----------------
if st.button("Process Text"):

    if text.strip() == "":
        st.warning("Please enter some text.")
        st.stop()

    # ---------------- TOKENIZATION ----------------
    if option == "Tokenization":
        st.subheader("Tokenization Output")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Sentence Tokenization")
            st.write(sent_tokenize(text))

        with col2:
            st.markdown("### Word Tokenization")
            st.write(word_tokenize(text))

        with col3:
            st.markdown("### Character Tokenization")
            st.write(list(text))

    # ---------------- TEXT CLEANING ----------------
    elif option == "Text Cleaning (Regex)":
        st.subheader("Regex-Based Text Cleaning")

        tokens = regex_preprocess(text)

        st.markdown("### Original Text")
        st.write(text)

        st.markdown("### Cleaned Tokens")
        st.write(tokens)

        st.markdown("### Cleaned Text")
        st.write(" ".join(tokens))

    # ---------------- STEMMING ----------------
    elif option == "Stemming":
        st.subheader("Stemming Output")

        words = word_tokenize(text)
        porter = PorterStemmer()
        lancaster = LancasterStemmer()

        df = pd.DataFrame({
            "Original Word": words,
            "Porter Stemmer": [porter.stem(w) for w in words],
            "Lancaster Stemmer": [lancaster.stem(w) for w in words]
        })

        st.dataframe(df, use_container_width=True)

    # ---------------- LEMMATIZATION ----------------
    elif option == "Lemmatization":
        st.subheader("Lemmatization using SpaCy")

        doc = nlp(text)
        data = [(token.text, token.pos_, token.lemma_) for token in doc]

        df = pd.DataFrame(data, columns=["Word", "POS", "Lemma"])
        st.dataframe(df, use_container_width=True)

    # ---------------- BAG OF WORDS ----------------
    elif option == "Bag of Words (Regex)":
        st.subheader("Bag of Words (Regex-Based)")

        tokens = regex_preprocess(text)
        cleaned_text = " ".join(tokens)

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform([cleaned_text])

        df = pd.DataFrame({
            "Word": vectorizer.get_feature_names_out(),
            "Frequency": X.toarray()[0]
        }).sort_values(by="Frequency", ascending=False)

        st.dataframe(df, use_container_width=True)

        df_top = df.head(10)
        fig, ax = plt.subplots()
        ax.bar(df_top["Word"], df_top["Frequency"])
        ax.set_title("Bag of Words – Top 10 Words")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # ---------------- TF-IDF (REGEX + PIE CHART) ----------------
    elif option == "TF-IDF (Regex)":
        st.subheader("TF-IDF using Regex-Based Preprocessing")

        tokens = regex_preprocess(text)
        cleaned_text = " ".join(tokens)

        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform([cleaned_text])

        df = pd.DataFrame({
            "Word": tfidf.get_feature_names_out(),
            "TF-IDF Score": X.toarray()[0]
        }).sort_values(by="TF-IDF Score", ascending=False)

        st.dataframe(df, use_container_width=True)

        df_top = df.head(10)

        # -------- BAR CHART --------
        st.markdown("### TF-IDF Bar Chart (Top 10 Words)")
        fig1, ax1 = plt.subplots()
        ax1.bar(df_top["Word"], df_top["TF-IDF Score"])
        ax1.set_xlabel("Words")
        ax1.set_ylabel("TF-IDF Score")
        ax1.set_title("TF-IDF – Top 10 Important Words")
        plt.xticks(rotation=45)
        st.pyplot(fig1)

        # -------- PIE CHART --------
        st.markdown("### TF-IDF Pie Chart (Top 10 Words)")
        fig2, ax2 = plt.subplots()
        ax2.pie(
            df_top["TF-IDF Score"],
            labels=df_top["Word"],
            autopct="%1.1f%%",
            startangle=90
        )
        ax2.axis("equal")
        st.pyplot(fig2)



