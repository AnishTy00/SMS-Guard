"""Unit and regression tests for Phase 3 preprocessing and dataset hygiene."""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.preprocessing import preprocess_text, NEGATION_WORDS


def test_negation_preservation():
    """Verify that negations are strictly preserved and not discarded."""
    sample = "I did not win any prize and do not want calls."
    processed = preprocess_text(sample)
    assert (
        "not" in processed.split()
    ), f"'not' was lost in preprocessing: {processed}"

    sample2 = "I can't and won't attend the meeting."
    processed2 = preprocess_text(sample2)
    # can't -> cannot, won't -> will not
    words = processed2.split()
    assert (
        "cannot" in words or "not" in words
    ), f"Negation lost in contraction: {processed2}"
    print("[PASS] Negation preservation test passed.")


def test_token_normalization():
    """Verify semantic tokens for URLs, currency, and numbers."""
    sample = "Win £1000 cash or $500! Visit https://win-scam.com or text 88888 now."
    processed = preprocess_text(sample)
    assert (
        "currency_token" in processed
    ), f"Currency token missing: {processed}"
    assert "url_token" in processed, f"URL token missing: {processed}"
    assert "number_token" in processed, f"Number token missing: {processed}"
    print("[PASS] Semantic token normalization test passed.")


def test_edge_cases():
    """Verify robust handling of empty, whitespace, and non-alphanumeric inputs."""
    assert preprocess_text("") == ""
    assert preprocess_text("   ") == ""
    assert preprocess_text("!!!???$$$") == "currency_token"
    assert preprocess_text(None) == ""
    assert preprocess_text(12345) == ""
    print("[PASS] Edge cases and invalid input test passed.")


def test_data_leakage_and_overlap():
    """Verify that train and test sets have zero overlapping messages."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "dataset", "train.csv")
    test_path = os.path.join(base_dir, "dataset", "test.csv")

    assert os.path.exists(train_path), "train.csv not found!"
    assert os.path.exists(test_path), "test.csv not found!"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_msgs = set(train_df["message"])
    test_msgs = set(test_df["message"])

    overlap = train_msgs.intersection(test_msgs)
    assert (
        len(overlap) == 0
    ), f"Data leakage detected! {len(overlap)} messages exist in both train and test sets."
    print(
        f"[PASS] Zero data leakage verified across {len(train_df)} train and {len(test_df)} test samples."
    )


if __name__ == "__main__":
    test_negation_preservation()
    test_token_normalization()
    test_edge_cases()
    test_data_leakage_and_overlap()
    print("\nALL PREPROCESSING TESTS PASSED SUCCESSFULLY!")
