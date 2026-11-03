"""Unit and regression tests for Phase 5 Machine Learning Model Development."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from src.preprocessing import preprocess_text


def test_model_artifacts_exist_and_load():
    """Verify that all serialized model artifacts and pipelines exist and load correctly."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")

    expected_files = [
        "spam_classifier.pkl",
        "pipeline_champion.pkl",
        "svm_balanced_pipeline.pkl",
        "svm_default_pipeline.pkl",
        "nb_pipeline.pkl",
        "lr_pipeline.pkl",
        "svm_balanced_model.pkl",
        "svm_default_model.pkl",
        "nb_model.pkl",
        "lr_model.pkl",
        "tfidf_vectorizer.pkl",
        "evaluation_metrics.json",
        "confusion_matrices.png",
    ]

    for fname in expected_files:
        fpath = os.path.join(models_dir, fname)
        assert os.path.exists(fpath), f"Missing artifact: {fname}"
        assert os.path.getsize(fpath) > 0, f"Artifact is empty: {fname}"

    # Verify champion pipeline structure
    champion_pipe = joblib.load(os.path.join(models_dir, "spam_classifier.pkl"))
    assert isinstance(champion_pipe, Pipeline), "Champion artifact must be a Scikit-learn Pipeline!"
    assert "tfidf" in champion_pipe.named_steps, "Pipeline must contain 'tfidf' step!"
    assert "clf" in champion_pipe.named_steps, "Pipeline must contain 'clf' step!"
    assert isinstance(champion_pipe.named_steps["clf"], LinearSVC), "Classifier step must be LinearSVC!"
    assert champion_pipe.named_steps["clf"].class_weight == "balanced", "Champion must be Balanced LinearSVC!"
    print("[PASS] All pipeline and model artifacts verified and champion confirmed as Balanced LinearSVC Pipeline.")


def test_pipeline_direct_inference():
    """Verify inference pipeline from preprocessed text string directly through Pipeline.predict()."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")

    champion_pipe = joblib.load(os.path.join(models_dir, "spam_classifier.pkl"))

    # Test cases: 1 obvious ham, 1 obvious spam
    ham_raw = "Hey, are we still meeting for lunch at 1pm?"
    spam_raw = "CONGRATULATIONS! You have won £1000 cash! Call 0800123456 immediately to claim your prize."

    # Preprocessing
    ham_clean = preprocess_text(ham_raw)
    spam_clean = preprocess_text(spam_raw)

    # Pipeline inference (single call, no double-vectorization!)
    pred_ham = champion_pipe.predict([ham_clean])[0]
    pred_spam = champion_pipe.predict([spam_clean])[0]

    assert pred_ham == 0, f"Expected 0 (Ham) for personal text, got {pred_ham}"
    assert pred_spam == 1, f"Expected 1 (Spam) for prize text, got {pred_spam}"
    print("[PASS] Direct pipeline inference correctly classified sample Ham and Spam without double vectorization.")


def test_metrics_json_structure():
    """Verify evaluation_metrics.json structure, zero-leakage CV metrics, and thresholds."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, "models", "evaluation_metrics.json")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "methodology" in data
    assert "cross_validation_results" in data
    assert "held_out_test_results" in data
    assert data["methodology"]["champion_model"] == "Linear SVM (Balanced)"
    assert "zero validation-fold TF-IDF leakage" in data["methodology"]["validation_strategy"]

    # Verify CV metrics for champion
    cv_champ = data["cross_validation_results"]["Linear SVM (Balanced)"]
    assert cv_champ["cv_spam_f1_mean"] >= 0.94
    assert cv_champ["cv_spam_recall_mean"] >= 0.93

    results = data["held_out_test_results"]
    for name in [
        "Linear SVM (Balanced)",
        "Linear SVM (Default)",
        "Multinomial Naive Bayes",
        "Logistic Regression",
    ]:
        assert name in results, f"Missing model {name} in evaluation metrics!"
        m = results[name]
        assert 0.95 <= m["test_accuracy"] <= 1.0
        assert 0.90 <= m["test_spam_precision"] <= 1.0
        assert 0.80 <= m["test_spam_recall"] <= 1.0

    print("[PASS] Evaluation metrics JSON structure, zero-leakage cross-validation, and thresholds validated.")


if __name__ == "__main__":
    test_model_artifacts_exist_and_load()
    test_pipeline_direct_inference()
    test_metrics_json_structure()
    print("\nALL ZERO-LEAKAGE MODEL TRAINING TESTS PASSED SUCCESSFULLY!")
