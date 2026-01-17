import nltk 
from nltk.tokenize import word_tokenize,sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
text = "Hello everyone 😊. Welcome to the world of Natural Language Processing ."

print("Word Tokens:")
print(word_tokenize(text))

print("\nSentence Tokens:")
print(sent_tokenize(text)) 