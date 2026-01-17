import streamlit as st
import re
st.title("🔍URL and Email Cleaner")

text = st.text_area("Enter your text with URLs and emails:")
remove_url = st.checkbox("Remove URLs")
remove_email = st.checkbox("Remove Emails")

if st.button("Clean Text"):
    result = text
    if remove_url:
        result = re.sub(r"http\S+|www\S+", "", result)

    if remove_email:
        result = re.sub(r"\S+@\S+", "", result)

    st.subheader("Cleaned Text:")
    st.write(result)