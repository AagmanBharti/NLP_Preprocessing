import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")

st.title("Text Tokenizer App")
text = st.text_area("Enter your text:")
if st.button("Analyze"):
    if text.strip():
        sentences = sent_tokenize(text)
        words = word_tokenize(text)

        st.write(" -- Results --")
        st.write("Sentence Count:",len(sentences))
        st.write("First Word: ",words[0] if words else "")
        st.write("Last Word: ",words[-1] if words else "")

        st.write("--Sentences--")
        for i,sent in enumerate(sentences,start=1):
            st.write(f"{i}. {sent}")
    else:
        st.warning("Please enter some text to analyze.")