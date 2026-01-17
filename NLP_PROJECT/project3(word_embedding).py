import streamlit as st
import nltk
import spacy
import string
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

# NLTK DOWNLOAD
nltk.download('punkt')

# LOAD SPACY MODEL
nlp = spacy.load("en_core_web_sm")

# REGEX PREPROCESSING FUNCTION 
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

    # Extract only alphabetic words
    tokens = re.findall(r'\b[a-z]+\b', text)

    return tokens

# STREAMLIT CONFIGURATION-
st.set_page_config(page_title="NLP Preprocessing", layout="wide")
st.title("NLP Text Preprocessing Application")
st.write("Tokenization, Cleaning, Stemming, Lemmatization, BoW, TF-IDF, Word Embeddings")

#  USER INPUT
text = st.text_area(
    "Enter Text for NLP Processing",
    height=150,
    placeholder="Example: Contact me at hello@gmail.com. Visit https://nlp.ai"
)

#  SIDEBAR 
option = st.sidebar.radio(
    "Select NLP Technique",
    [
        "Tokenization",
        "Text Cleaning",
        "Stemming",
        "Lemmatization",
        "Bag of Words",
        "TF-IDF",
        "Word Embeddings (Word2Vec)"
    ]
)

# PROCESS BUTTON 
if st.button("Process Text"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    # TOKENIZATION 
    elif option == "Tokenization":
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

    # TEXT CLEANING (REGEX) 
    elif option == "Text Cleaning":
        st.subheader("Regex-Based Text Cleaning")

        tokens = regex_preprocess(text)

        st.markdown("### Original Text")
        st.write(text)

        st.markdown("### Cleaned Tokens")
        st.write(tokens)

        st.markdown("### Cleaned Text")
        st.write(" ".join(tokens))

    # STEMMING
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

    # LEMMATIZATION
    elif option == "Lemmatization":
        st.subheader("Lemmatization using SpaCy")

        doc = nlp(text) 
        data = [(token.text, token.pos_, token.lemma_) for token in doc]

        df = pd.DataFrame(data, columns=["Word", "POS", "Lemma"])
        st.dataframe(df, use_container_width=True)

    # BAG OF WORDS (REGEX)-
    elif option == "Bag of Words":
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
        ax.set_title("BoW - Top 10 Words") 
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # TF-IDF (REGEX) 
    elif option == "TF-IDF":
        st.subheader("TF-IDF (Regex-Based)")

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
        fig, ax = plt.subplots()
        ax.bar(df_top["Word"], df_top["TF-IDF Score"])
        ax.set_title("TF-IDF - Top 10 Words")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # WORD2VEC (REGEX)
    elif option == "Word Embeddings (Word2Vec)":
        st.subheader("Word Embeddings using Word2Vec (Regex-Based)")

        tokens = regex_preprocess(text)

        if len(tokens) < 2:
            st.warning("Please enter more meaningful text.")
        else:
            model = Word2Vec(
                sentences=[tokens],
                vector_size=100,
                window=5,
                min_count=1,
                workers=4
            )

            words = list(model.wv.index_to_key)
            vectors = np.array([model.wv[word] for word in words])

            st.markdown("### Vector Information")
            df_vec = pd.DataFrame({
                "Word": words,
                "Vector Size": [model.vector_size] * len(words)
            })
            st.dataframe(df_vec, use_container_width=True)

            st.markdown("### Word Similarity Matrix")
            sim = cosine_similarity(vectors)
            df_sim = pd.DataFrame(sim, index=words, columns=words)

            st.dataframe(
                df_sim.style.background_gradient(cmap="Blues"),
                use_container_width=True
            )

            selected_word = st.selectbox("Select a word", words)
            similar = model.wv.most_similar(selected_word, topn=5)

            df_sim_words = pd.DataFrame(similar, columns=["Word", "Similarity Score"])
            st.dataframe(df_sim_words, use_container_width=True)

            fig, ax = plt.subplots()
            ax.bar(df_sim_words["Word"], df_sim_words["Similarity Score"])
            ax.set_ylim(0, 1)
            ax.set_title(f"Top Similar Words to '{selected_word}'")
            st.pyplot(fig)
