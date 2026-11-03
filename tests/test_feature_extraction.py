"""Unit and regression tests for Phase 4 TF-IDF feature extraction."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from src.feature_extraction import (
    load_vectorizer,
    build_pipeline,
    extract_features,
)
from src.preprocessing import preprocess_text


def test_vectorizer_artifact():
    """Verify that the persisted vectorizer artifact loads and has valid vocabulary."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    vectorizer = load_vectorizer(models_dir)

    vocab = vectorizer.get_feature_names_out()
    assert len(vocab) == 4000, f"Expected 4000 features, got {len(vocab)}"
    assert "not" in vocab, "'not' should be in the TF-IDF vocabulary"
    assert "no" in vocab, "'no' should be in the TF-IDF vocabulary"
    assert "url_token" in vocab, "'url_token' should be in vocabulary"
    assert "number_token" in vocab, "'number_token' should be in vocabulary"
    print("[PASS] Vectorizer artifact loading and vocabulary verified.")


def test_matrix_properties():
    """Verify non-negativity and matrix dimensions."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_csv = os.path.join(base_dir, "dataset", "train.csv")
    test_csv = os.path.join(base_dir, "dataset", "test.csv")
    models_dir = os.path.join(base_dir, "models")

    res = extract_features(train_csv, test_csv, models_dir)
    X_train = res["X_train_tfidf"]
    X_test = res["X_test_tfidf"]

    assert X_train.shape == (
        4136,
        4000,
    ), f"Unexpected train shape: {X_train.shape}"
    assert X_test.shape == (1035, 4000), f"Unexpected test shape: {X_test.shape}"

    # Non-negativity check (required for MultinomialNB)
    assert (
        X_train.data.min() >= 0.0
    ), "TF-IDF contains negative values (invalid for MultinomialNB)!"
    assert (
        X_test.data.min() >= 0.0
    ), "Test TF-IDF contains negative values (invalid for MultinomialNB)!"
    print(
        "[PASS] Feature matrices dimensions and non-negativity verified (MultinomialNB & LinearSVC compatible)."
    )


def test_pipeline_integration():
    """Verify that Scikit-learn Pipeline operates seamlessly with LinearSVC."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_df = pd.read_csv(os.path.join(base_dir, "dataset", "train.csv"))

    pipeline = build_pipeline(LinearSVC(random_state=42))
    # Quick fit on a small slice to ensure pipeline mechanics
    subset_df = train_df.head(100)
    pipeline.fit(subset_df["cleaned_message"], subset_df["label_num"])

    sample_msg = preprocess_text("Congratulations! You won $1000 call now!")
    pred = pipeline.predict([sample_msg])
    assert pred[0] in [0, 1], f"Unexpected prediction value: {pred[0]}"
    print("[PASS] Scikit-learn Pipeline with LinearSVC integration verified.")


if __name__ == "__main__":
    test_vectorizer_artifact()
    test_matrix_properties()
    test_pipeline_integration()
    print("\nALL FEATURE EXTRACTION TESTS PASSED SUCCESSFULLY!")
