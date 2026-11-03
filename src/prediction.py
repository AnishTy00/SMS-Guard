"""SMS Message Inference and Risk Integration Pipeline.

Connects the saved Linear SVM champion pipeline with preprocessing
and the heuristic risk engine for real-time inference.
"""

import math
import os
import sys
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
from src.preprocessing import preprocess_text
from src.risk_engine import analyze_risk_indicators

# Global cached pipeline singleton
_CHAMPION_PIPELINE: Optional[Any] = None


def load_classifier(model_path: Optional[str] = None) -> Any:
    """Load the persisted Scikit-learn Linear SVM champion pipeline."""
    global _CHAMPION_PIPELINE
    if model_path is None and _CHAMPION_PIPELINE is not None:
        return _CHAMPION_PIPELINE

    target_path = model_path
    if target_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        target_path = os.path.join(base_dir, "models", "spam_classifier.pkl")

    if not os.path.exists(target_path):
        raise FileNotFoundError(
            f"Champion model pipeline not found at {target_path}. Please execute Phase 5 training first."
        )

    pipeline = joblib.load(target_path)
    if model_path is None:
        _CHAMPION_PIPELINE = pipeline
    return pipeline


def predict_message(raw_text: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    """Perform end-to-end classification and risk analysis on an SMS message.

    Workflow:
    1. Input validation.
    2. Text preprocessing and normalization via src.preprocessing.
    3. Pipeline inference (TF-IDF vectorizer + Linear SVM classifier in a single pass).
    4. Continuous margin distance via decision_function().
    5. Heuristic confidence estimation via monotonic sigmoid score transformation.
    6. Heuristic scam/risk indicator scanning via src.risk_engine.

    Returns:
        Structured prediction result dictionary.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return {
            "is_valid": False,
            "error_message": "Message is empty or contains only whitespace. Please enter valid text.",
            "label": None,
            "is_spam": False,
            "confidence_pct": 0.0,
            "decision_score": 0.0,
            "cleaned_text": "",
            "risk_analysis": analyze_risk_indicators(""),
        }

    pipeline = load_classifier(model_path)

    # Step 1: Preprocess message consistently with training data
    cleaned = preprocess_text(raw_text)

    # Handle case where all tokens were removed (e.g., punctuation only)
    effective_input = cleaned if cleaned.strip() else raw_text.lower()

    # Step 2: Atomic Pipeline Inference (avoids double TF-IDF vectorization)
    pred_num = int(pipeline.predict([effective_input])[0])
    label = "Spam" if pred_num == 1 else "Ham"

    # Step 3: Continuous Decision Function (signed distance from separating hyperplane)
    score = float(pipeline.decision_function([effective_input])[0])

    # Step 4: Confidence estimation via monotonic sigmoid transformation: 1 / (1 + exp(-score))
    # Note: This is an uncalibrated monotonic squashing of the decision margin, not a fitted Platt scaling probability.
    # For Balanced Linear SVM:
    # If score > 0 -> Spam likelihood > 50%
    # If score < 0 -> Spam likelihood < 50% (Ham)
    try:
        spam_prob = 1.0 / (1.0 + math.exp(-score))
    except OverflowError:
        spam_prob = 1.0 if score > 0 else 0.0

    # Class-specific confidence
    if pred_num == 1:
        confidence = spam_prob
    else:
        confidence = 1.0 - spam_prob

    # Step 5: Heuristic Risk Signal Detection
    risk_analysis = analyze_risk_indicators(raw_text)

    return {
        "is_valid": True,
        "error_message": None,
        "label": label,
        "is_spam": (pred_num == 1),
        "confidence_score": round(confidence, 4),
        "confidence_pct": round(confidence * 100, 2),
        "decision_score": round(score, 4),
        "cleaned_text": cleaned,
        "raw_text": raw_text,
        "risk_analysis": risk_analysis,
    }


if __name__ == "__main__":
    test_cases = [
        "Hey Anish, are we meeting at the library at 4pm?",
        "URGENT! Your bank account has been suspended. Verify your OTP at http://secure-bank.xyz immediately or call 0800123456.",
        "Congratulations! You won a $1,000 Walmart Gift Card! Claim now: bit.ly/claim1000",
        "Can't make it tonight, sorry!",
    ]

    print("Testing SMS Guard Prediction Engine:\n")
    for msg in test_cases:
        res = predict_message(msg)
        print(f"Text: '{msg[:70]}...'")
        print(f" -> Label: {res['label']} (Confidence: {res['confidence_pct']}%, Margin: {res['decision_score']})")
        print(f" -> Risk Tier: {res['risk_analysis']['risk_tier']} ({res['risk_analysis']['indicator_count']} flags)")
        print()
