from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
corpus = ["text one", "text two", "text three"] 
vector = TfidfVectorizer()
vector1 = CountVectorizer()  
X = vector.fit_transform(corpus) 
Y = vector1.fit_transform(corpus) 
print(Y.toarray()) 
print(vector1.get_feature_names_out()) 
print(Y)
print(X.toarray()) 
print(vector.get_feature_names_out()) 
print(X)