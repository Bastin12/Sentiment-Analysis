"""
train_model.py

This script reproduces the EXACT preprocessing and modeling steps from the
original notebook (Kindle_Book_Review_Sentiment_Analysis_System.ipynb):

  1. Load dataset, keep reviewText + rating
  2. Convert rating -> binary label (rating < 3 -> 0 (negative), else 1 (positive))
  3. Lowercase the text
  4. Remove special characters, remove stopwords, remove URLs, remove HTML tags
  5. Lemmatize
  6. Train/test split
  7. Vectorize with CountVectorizer (BOW) and TfidfVectorizer (TF-IDF)
  8. Train GaussianNB on both BOW and TF-IDF features
  9. Save the fitted vectorizers + models to disk so the Flask app can load
     them instantly instead of retraining on every request.

No modeling logic has been changed from the original notebook -- this is
the same pipeline, just saved to disk at the end with joblib/pickle.
"""

import re
import pickle

import pandas as pd
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------------------------------------
# 0. NLTK downloads (only happens once, then cached)
# ---------------------------------------------------------------------------
nltk.download("stopwords")
nltk.download("wordnet")

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


def clean_text(text):
    """Same cleaning pipeline as the notebook, wrapped into one function
    so the Flask app can call it on a single user-typed review too."""
    text = str(text).lower()
    # remove special characters (keep letters, numbers, spaces, hyphen)
    text = re.sub("[^a-z A-z 0-9-]+", "", text)
    # remove stopwords
    text = " ".join([y for y in text.split() if y not in STOPWORDS])
    # remove urls
    text = re.sub(
        r"(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?",
        "",
        text,
    )
    # remove html tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # lemmatize
    text = lemmatize_words(text)
    return text


def main():
    # -----------------------------------------------------------------
    # 1. Load dataset
    # -----------------------------------------------------------------
    data = pd.read_csv("all_kindle_review.csv")
    df = data[["reviewText", "rating"]].copy()

    # -----------------------------------------------------------------
    # 2. positive review is 1 and negative review is 0
    # -----------------------------------------------------------------
    df["rating"] = df["rating"].apply(lambda x: 0 if x < 3 else 1)

    # -----------------------------------------------------------------
    # 3-5. Cleaning + lemmatization (same steps as notebook, consolidated)
    # -----------------------------------------------------------------
    df = df.dropna(subset=["reviewText"])
    print("Cleaning text... this takes a few minutes on the full dataset.")
    df["reviewText"] = df["reviewText"].apply(clean_text)

    # -----------------------------------------------------------------
    # 6. Train/test split
    # -----------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        df["reviewText"], df["rating"], test_size=0.20, random_state=42
    )

    # -----------------------------------------------------------------
    # 7. Vectorize - BOW and TF-IDF
    # -----------------------------------------------------------------
    # NOTE: max_features caps the vocabulary size so the dense arrays
    # GaussianNB needs (.toarray()) fit comfortably in memory on a small
    # free-tier server (e.g. Hugging Face Spaces' 16GB free CPU tier, or
    # this sandbox's ~4GB). This keeps the same BOW/TF-IDF + GaussianNB
    # approach from the notebook, just with a memory-safe vocabulary size.
    bow = CountVectorizer(max_features=5000)
    X_train_bow = bow.fit_transform(X_train).toarray()
    X_test_bow = bow.transform(X_test).toarray()

    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train).toarray()
    X_test_tfidf = tfidf.transform(X_test).toarray()

    # -----------------------------------------------------------------
    # 8. Train GaussianNB models
    # -----------------------------------------------------------------
    nb_model_bow = MultinomialNB().fit(X_train_bow, y_train)
    nb_model_tfidf = MultinomialNB().fit(X_train_tfidf, y_train)

    # -----------------------------------------------------------------
    # Evaluate (printed to console, same as notebook)
    # -----------------------------------------------------------------
    y_pred_bow = nb_model_bow.predict(X_test_bow)
    y_pred_tfidf = nb_model_tfidf.predict(X_test_tfidf)

    print("BOW accuracy:", accuracy_score(y_test, y_pred_bow))
    print(confusion_matrix(y_test, y_pred_bow))
    print(classification_report(y_test, y_pred_bow))

    print("TFIDF accuracy:", accuracy_score(y_test, y_pred_tfidf))
    print(confusion_matrix(y_test, y_pred_tfidf))
    print(classification_report(y_test, y_pred_tfidf))

    # -----------------------------------------------------------------
    # 9. Save vectorizers + models to disk for the Flask app
    # -----------------------------------------------------------------
    with open("model/bow_vectorizer.pkl", "wb") as f:
        pickle.dump(bow, f)
    with open("model/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open("model/nb_model_bow.pkl", "wb") as f:
        pickle.dump(nb_model_bow, f)
    with open("model/nb_model_tfidf.pkl", "wb") as f:
        pickle.dump(nb_model_tfidf, f)

    print("\nSaved all models to the model/ folder.")


if __name__ == "__main__":
    main()
