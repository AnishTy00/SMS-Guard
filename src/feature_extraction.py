"""TF-IDF feature extraction module.

Enforces strict zero-leakage protocols:
1. Fits TfidfVectorizer exclusively on X_train.
2. Transforms X_test strictly using the fitted vectorizer.
3. Non-negative TF-IDF outputs ensure universal compatibility across:
   - Module 5: Linear Support Vector Classifier (LinearSVC)
   - Baseline 1: Multinomial Naive Bayes (MultinomialNB)
   - Baseline 2: Logistic Regression (LogisticRegression)
4. Provides reusable Scikit-learn Pipeline constructor to prevent leakage during cross-validation.
5. Serializes fitted vectorizer artifact using Joblib.
"""

import json
import os
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def get_tfidf_vectorizer(
    max_features: int = 4000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Instantiate a TF-IDF Vectorizer configured for SMS text classification.

    Parameters:
        max_features: Maximum vocabulary size (default 4000).
        ngram_range: Range of n-grams (1, 2 captures unigrams and bigrams).
        min_df: Minimum document frequency threshold to eliminate single-occurrence noise.
        sublinear_tf: Applies sublinear scaling (1 + log(tf)) to dampen word frequency scaling.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )


def build_pipeline(classifier: Any, **vectorizer_kwargs) -> Pipeline:
    """Build a Scikit-learn Pipeline binding TF-IDF vectorization with any classifier.

    Ensures that vectorization and classification are atomic, eliminating any potential
    for feature leakage during training, cross-validation, and production inference.
    """
    vectorizer = get_tfidf_vectorizer(**vectorizer_kwargs)
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def extract_features(
    train_csv_path: str,
    test_csv_path: str,
    output_model_dir: str,
    max_features: int = 4000,
) -> Dict[str, Any]:
    """Fit TF-IDF vectorizer exclusively on training data and transform test data.

    Returns:
        Dictionary containing vectorizer, feature matrices, shapes, and metadata summary.
    """
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Handle any potential empty string / NaN values safely
    X_train_text = train_df["cleaned_message"].fillna("").astype(str)
    y_train = train_df["label_num"].values

    X_test_text = test_df["cleaned_message"].fillna("").astype(str)
    y_test = test_df["label_num"].values

    # Step 1: Fit TF-IDF strictly on training text only
    vectorizer = get_tfidf_vectorizer(max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train_text)

    # Step 2: Transform test text using fitted vocabulary (NO fitting on test)
    X_test_tfidf = vectorizer.transform(X_test_text)

    # Step 3: Serialize vectorizer artifact
    os.makedirs(output_model_dir, exist_ok=True)
    vectorizer_path = os.path.join(output_model_dir, "tfidf_vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_path)

    # Step 4: Extract diagnostic vocabulary statistics
    feature_names = vectorizer.get_feature_names_out()
    vocab_size = len(feature_names)

    # Compute top TF-IDF features by mean score in training corpus
    mean_tfidf_scores = np.array(X_train_tfidf.mean(axis=0)).flatten()
    top_indices = mean_tfidf_scores.argsort()[::-1][:25]
    top_features = [
        {"feature": feature_names[i], "mean_tfidf": round(float(mean_tfidf_scores[i]), 5)}
        for i in top_indices
    ]

    summary = {
        "vectorizer_artifact_path": vectorizer_path,
        "max_features_configured": max_features,
        "actual_vocabulary_size": vocab_size,
        "ngram_range": list(vectorizer.ngram_range),
        "sublinear_tf": vectorizer.sublinear_tf,
        "train_matrix_shape": list(X_train_tfidf.shape),
        "test_matrix_shape": list(X_test_tfidf.shape),
        "train_samples": int(X_train_tfidf.shape[0]),
        "test_samples": int(X_test_tfidf.shape[0]),
        "feature_count": int(X_train_tfidf.shape[1]),
        "compatibility": {
            "LinearSVC": "Compatible (Sparse matrix, non-negative features)",
            "MultinomialNB": "Compatible (Non-negative TF-IDF weights)",
            "LogisticRegression": "Compatible (High-dimensional sparse input)",
        },
        "top_25_features": top_features,
    }

    summary_path = os.path.join(output_model_dir, "tfidf_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "vectorizer": vectorizer,
        "X_train_tfidf": X_train_tfidf,
        "y_train": y_train,
        "X_test_tfidf": X_test_tfidf,
        "y_test": y_test,
        "summary": summary,
    }


def load_vectorizer(model_dir: str) -> TfidfVectorizer:
    """Load the persisted TF-IDF vectorizer artifact from disk."""
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"TF-IDF vectorizer artifact not found at: {vectorizer_path}")
    return joblib.load(vectorizer_path)


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_csv = os.path.join(base, "dataset", "train.csv")
    test_csv = os.path.join(base, "dataset", "test.csv")
    models_dir = os.path.join(base, "models")

    print("Executing Phase 4: TF-IDF Feature Extraction...")
    res = extract_features(train_csv, test_csv, models_dir)
    sum_data = res["summary"]

    print("\n=== TF-IDF FEATURE EXTRACTION COMPLETED ===")
    print(f"Vectorizer saved to: {sum_data['vectorizer_artifact_path']}")
    print(f"Vocabulary Size: {sum_data['actual_vocabulary_size']}")
    print(f"Train Matrix Shape: {sum_data['train_matrix_shape']}")
    print(f"Test Matrix Shape: {sum_data['test_matrix_shape']}")
    print("\nTop 10 Informative Features by Mean TF-IDF:")
    for f in sum_data["top_25_features"][:10]:
        print(f" - {f['feature']}: {f['mean_tfidf']}")
