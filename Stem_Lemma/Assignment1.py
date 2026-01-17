import streamlit as st
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
from nltk.stem import PorterStemmer

nltk.download('punkt')

st.title("💭 Root Word Cloud Generator")
text = st.text_area("Enter your text:")
method = st.radio("Choose method:",["Stemming","Lemmatization"])

if text:
    words = nltk.word_tokenize(text)
    ps = PorterStemmer()
    nlp = spacy.load("en_core_web_sm")  
    roots = []
    for word in words:
        if word.isalpha():
            if method == "Stemming":
                roots.append(ps.stem(word.lower()))
            else:
                roots.append(nlp(word.lower())[0].lemma_)
    freq = "".join(roots)
    wc = WordCloud(width=600, height=300, background_color='white').generate(freq)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')

    st.pyplot(fig)




