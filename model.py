from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Sample data
X_train = ["happy", "sad", "excited", "bored", "lonely"]
y_train = ["Happy Gilmore", "The Notebook", "The Dark Knight", "The Endless Summer", "How to lose a guy in 10 days"]

# Vectorize and train model
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Save model and vectorizer
with open("movie_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)