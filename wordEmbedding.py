import gensim.downloader as api 
import numpy as np
model = api.load("word2vec-google-news-300")
text = "machine learning is great"
words = [ 
word for word in text.lower().split() 
if word in model.key_to_index 
] 
vector = np.mean([model[word] for word in words], axis=0) 
print("Shape of vector:", vector.shape) 