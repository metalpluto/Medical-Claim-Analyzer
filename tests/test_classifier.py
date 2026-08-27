"""
tests/test_classifier.py

Unit tests for the TF-IDF + Logistic Regression denial classifier.
Training and prediction both run fully offline against the local
dataset in data/denial_letters.json, so these tests need no API key
and no network access.

A model is trained once per test session (see the `trained_model`
fixture) rather than reloading/retraining per test, to keep the
suite fast.
"""

import pytest

from src.classifier import build_pipeline, load_dataset, predict, train

VALID_CATEGORIES = {
    "medical_necessity",
    "missing_authorization",
    "coding_error",
    "timely_filing",
    "eligibility_issue",
    "duplicate_claim",
}


@pytest.fixture(scope="module")
def trained_model():
    # save False: don't overwrite the committed model artifact just for tests
    return train(save=False)


def test_dataset_loads_and_is_labeled():
    texts, labels = load_dataset()
    assert len(texts) > 0
    assert len(texts) == len(labels)
    assert set(labels).issubset(VALID_CATEGORIES)


def test_build_pipeline_has_expected_steps():
    pipeline = build_pipeline()
    step_names = [name for name, _ in pipeline.steps]
    assert step_names == ["tfidf", "clf"]


def test_train_returns_fitted_pipeline(trained_model):
    # A fitted sklearn Pipeline can call predict without raising
    prediction = trained_model.predict(["The claim was denied for a coding error."])
    assert prediction[0] in VALID_CATEGORIES


def test_predict_returns_category_and_confidence(trained_model):
    text = (
        "The requested MRI (CPT 72148) does not meet clinical criteria "
        "for medical necessity per our coverage policy."
    )
    result = predict(text, model=trained_model)
    assert "category" in result
    assert "confidence" in result
    assert result["category"] in VALID_CATEGORIES
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_medical_necessity_example(trained_model):
    text = (
        "This procedure does not meet clinical criteria for medical "
        "necessity based on the submitted documentation."
    )
    result = predict(text, model=trained_model)
    assert result["category"] == "medical_necessity"


def test_predict_missing_authorization_example(trained_model):
    text = (
        "The requested procedure was performed without prior authorization "
        "on file. Please submit a retroactive authorization request."
    )
    result = predict(text, model=trained_model)
    assert result["category"] == "missing_authorization"


def test_predict_is_deterministic_for_same_input(trained_model):
    text = "Claim denied because it was submitted after the timely filing deadline."
    first = predict(text, model=trained_model)
    second = predict(text, model=trained_model)
    assert first == second
