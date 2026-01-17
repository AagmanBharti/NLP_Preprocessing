import spacy 
nlp = spacy.load("en_core_web_sm") 
doc = nlp("The children are running in the playground") 
lemmas = [token.lemma_ for token in doc] 
print(lemmas)