---
title: Kindle Review Sentiment Analyzer
emoji: 📚
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Kindle Book Review Sentiment Analyzer

An NLP project that classifies Amazon Kindle book reviews as **positive** or
**negative** using two classic text-vectorization approaches — **Bag of
Words** and **TF-IDF** — each feeding a **Gaussian Naive Bayes** classifier.

🔗 **Live demo:** this Space (see the URL in your browser's address bar once deployed)

## How it works

1. **Preprocessing** — every review is lowercased, stripped of special
   characters/URLs/HTML, has stopwords removed, and is lemmatized.
2. **Vectorization** — the cleaned text is converted into numeric features
   two ways: Bag of Words (`CountVectorizer`) and TF-IDF (`TfidfVectorizer`).
3. **Classification** — each vector representation is fed to its own
   `GaussianNB` model trained on 12,000 Kindle reviews.
4. **Demo** — type any review on the web page and see what both models
   predict, side by side, along with the model's confidence.

## Tech stack

- **Python**, **scikit-learn**, **NLTK**, **BeautifulSoup** — preprocessing & ML
- **Flask** — web server / API
- **Docker** — containerized deployment on Hugging Face Spaces

## Project structure

```
.
├── app.py                  # Flask app (loads saved models, serves predictions)
├── train_model.py          # Training script (same pipeline as the original notebook)
├── all_kindle_review.csv   # Dataset (12,000 Kindle book reviews)
├── requirements.txt
├── Dockerfile
├── model/                  # Saved vectorizers + trained models (.pkl)
│   ├── bow_vectorizer.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── nb_model_bow.pkl
│   └── nb_model_tfidf.pkl
└── templates/
    └── index.html           # Front-end page
```

## Run locally

```bash
pip install -r requirements.txt
python train_model.py     # only needed once, to regenerate the model/ folder
python app.py
# visit http://localhost:7860
```

## API

```bash
curl -X POST http://localhost:7860/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "I loved this book, could not put it down!"}'
```

Returns:
```json
{
  "cleaned_text": "loved book put",
  "bow": { "label": "Positive", "confidence": 87.3 },
  "tfidf": { "label": "Positive", "confidence": 91.2 }
}
```
