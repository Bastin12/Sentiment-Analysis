"""
app.py

Flask web app for the Kindle Book Review Sentiment Analysis project.

- Loads the BOW + TF-IDF vectorizers and GaussianNB models that were
  trained and saved by train_model.py (same pipeline as the original
  notebook -- nothing about the ML logic is changed here).
- Exposes a simple HTML page where a user types a review and gets back
  the prediction from BOTH models (BOW+NB and TF-IDF+NB).
- Exposes a JSON API endpoint (/predict) in case you want to call it
  from JavaScript, Postman, or another app later.
"""

import os
import re
import pickle

import nltk
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# NLTK setup (downloads only once, cached afterwards)
# ---------------------------------------------------------------------------
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")


def load_pickle(filename):
    with open(os.path.join(MODEL_DIR, filename), "rb") as f:
        return pickle.load(f)


# Load everything once at startup (not per-request) for speed.
bow_vectorizer = load_pickle("bow_vectorizer.pkl")
tfidf_vectorizer = load_pickle("tfidf_vectorizer.pkl")
nb_model_bow = load_pickle("nb_model_bow.pkl")
nb_model_tfidf = load_pickle("nb_model_tfidf.pkl")


def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


def clean_text(text):
    """Exact same cleaning pipeline used during training, applied to the
    user's typed review before vectorizing it."""
    text = str(text).lower()
    text = re.sub("[^a-z A-z 0-9-]+", "", text)
    text = " ".join([y for y in text.split() if y not in STOPWORDS])
    text = re.sub(
        r"(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?",
        "",
        text,
    )
    text = BeautifulSoup(text, "html.parser").get_text()
    text = lemmatize_words(text)
    return text


def predict_review(review_text):
    cleaned = clean_text(review_text)

    bow_vec = bow_vectorizer.transform([cleaned]).toarray()
    tfidf_vec = tfidf_vectorizer.transform([cleaned]).toarray()

    bow_pred = int(nb_model_bow.predict(bow_vec)[0])
    tfidf_pred = int(nb_model_tfidf.predict(tfidf_vec)[0])

    bow_proba = nb_model_bow.predict_proba(bow_vec)[0]
    tfidf_proba = nb_model_tfidf.predict_proba(tfidf_vec)[0]

    print("Cleaned:", cleaned)
    print("BOW Prediction:", bow_pred)
    print("BOW Probabilities:", bow_proba)
    print("TFIDF Prediction:", tfidf_pred)
    print("TFIDF Probabilities:", tfidf_proba)

    return {
        "cleaned_text": cleaned,
        "bow": {
            "label": "Positive" if bow_pred == 1 else "Negative",
            "confidence": round(float(max(bow_proba)) * 100, 2),
        },
        "tfidf": {
            "label": "Positive" if tfidf_pred == 1 else "Negative",
            "confidence": round(float(max(tfidf_proba)) * 100, 2),
        },
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    review_text = ""

    if request.method == "POST":
        review_text = request.form.get("review_text", "").strip()
        if review_text:
            result = predict_review(review_text)

    return render_template("index.html", result=result, review_text=review_text)


@app.route("/predict", methods=["POST"])
def predict_api():
    """JSON API: POST { "review": "..." } -> JSON predictions."""
    data = request.get_json(silent=True) or {}
    review_text = data.get("review", "").strip()

    if not review_text:
        return jsonify({"error": "Please provide a non-empty 'review' field."}), 400

    result = predict_review(review_text)
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
