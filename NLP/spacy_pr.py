import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

tokens_text = [token.text for token in doc]
print("Tokens:")
print(tokens_text)
print("Number of tokens:", len(tokens_text))

tokens_obj = [token for token in doc] 
print("\nToken Objects:")
print(tokens_obj)
print("Number of token objects:", len(tokens_obj))