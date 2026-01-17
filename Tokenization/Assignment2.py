import streamlit as st
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt_tab')

nltk.download('stopwords')
st.title("📝Stopword,punctuation and number Removal")
remove_punct = st.checkbox("Remove punctuation")
remove_num = st.checkbox("Remove numbers")

text = st.text_area("Enter your text:")
if st.button("Clean Text"):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered = [word for word in tokens if word.lower() not in stop_words]
    removed_count = len(tokens) - len(filtered)
    result = ' '.join(filtered)

    if remove_punct:
        result = ''.join([ch for ch in result if ch not in string.punctuation])
    if remove_num:
        result = ''.join([ch for ch in result if not ch.isdigit()])
    st.subheader("Text after Stopword,punctuation and number Removal:")
    st.write(result)
    st.subheader("Number of stopwords removed:")
    st.write(removed_count)

