from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
corpus = [
    "Cats are cute and fluffy",
    "Dogs are loyal and friendly",
    "Cats and dogs can be friends"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
for i,row in df.iterrows():
    print(f"Document {i+1}:")
    top_words = row.sort_values(ascending=False).head(2)
    print(top_words)
    print()
    