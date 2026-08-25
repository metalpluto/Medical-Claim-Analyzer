"""
classifier.py

Text classification for denial letters: given the free-text body of a
denial letter, predict which category it falls into.

Pipeline: TF-IDF vectorization -> Logistic Regression.
TF-IDF turns each letter into a vector of weighted word/n-gram
importance scores Logistic Regression then learns a decision boundary
between categories from those vectors. This is a classic, interpretable
NLP baseline — the natural next step from here is swapping the TF-IDF
vectorizer for sentence embeddings (e.g. from a transformer model) for
higher accuracy on more varied real-world text.

Categories:
    medical_necessity, missing_authorization, coding_error,
    timely_filing, eligibility_issue, duplicate_claim
"""

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = Path(__file__).parent.parent / "data" / "denial_letters.json"
MODEL_PATH = Path(__file__).parent.parent / "model" / "denial_classifier.joblib"


def load_dataset() -> tuple[list[str], list[str]]:
    with open(DATA_PATH, encoding="utf-8") as f:
        records = json.load(f)
    texts = [r["text"] for r in records]
    labels = [r["category"] for r in records]
    return texts, labels


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1,
        )),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def train(save: bool = True) -> Pipeline:
    texts, labels = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Held-out test accuracy: {acc:.0%}")
    print(classification_report(y_test, preds, zero_division=0))

    # Refit on the full dataset for the saved production model
    pipeline_full = build_pipeline()
    pipeline_full.fit(texts, labels)

    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(pipeline_full, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    return pipeline_full


def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        return train()
    return joblib.load(MODEL_PATH)


def predict(text: str, model: Pipeline | None = None) -> dict:
    model = model or load_model()
    category = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    confidence = max(proba)
    return {"category": category, "confidence": round(float(confidence), 3)}


if __name__ == "__main__":
    train()

    sample = (
        "The requested MRI (CPT 72148) does not meet clinical criteria "
        "for medical necessity per our coverage policy."
    )
    print("\nSample prediction:")
    print(predict(sample))
