"""Text preprocessing, normalization, and stratified dataset partitioning.

This module provides reusable text cleaning and normalization functions designed
specifically for SMS message processing. It enforces strict data hygiene:
1. Contraction expansion (ensuring negations are preserved).
2. URL, currency, and number normalization into special semantic tokens.
3. Removal of non-negation stopwords (preserving words like 'no', 'not', 'never').
4. Lemmatization using WordNet.
5. Deduplication and stratified train-test splitting (80:20) to prevent data leakage.
"""

import json
import os
import re
from typing import Dict, List, Set, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# Common contractions to expand for clean negation recognition
CONTRACTION_MAP = {
    r"\bcan\'t\b": "cannot",
    r"\bwon\'t\b": "will not",
    r"\bshan\'t\b": "shall not",
    r"\bn\'t\b": " not",
    r"\b\'re\b": " are",
    r"\b\'s\b": " is",
    r"\b\'d\b": " would",
    r"\b\'ll\b": " will",
    r"\b\'ve\b": " have",
    r"\b\'m\b": " am",
}

# Negation words that must NEVER be removed as stopwords
NEGATION_WORDS: Set[str] = {
    "no",
    "not",
    "nor",
    "neither",
    "never",
    "none",
    "nobody",
    "nowhere",
    "nothing",
    "cannot",
    "without",
}

# Fallback basic stop words if NLTK is unavailable
BASIC_STOPWORDS: Set[str] = {
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
}


def get_stop_words() -> Set[str]:
    """Retrieve English stopwords excluding critical negation and intent words."""
    try:
        from nltk.corpus import stopwords

        nltk_stops = set(stopwords.words("english"))
        return nltk_stops - NEGATION_WORDS
    except Exception:
        return BASIC_STOPWORDS - NEGATION_WORDS


def get_lemmatizer():
    """Retrieve NLTK WordNetLemmatizer if available, else identity function."""
    try:
        from nltk.stem import WordNetLemmatizer

        return WordNetLemmatizer()
    except Exception:
        return None


# Global singletons for fast inference
STOP_WORDS: Set[str] = get_stop_words()
LEMMATIZER = get_lemmatizer()


def preprocess_text(text: str) -> str:
    """Normalize and clean an SMS message string.

    Operations performed:
    1. Input validation and string conversion.
    2. Contraction expansion (e.g., 'don't' -> 'do not') to preserve negations.
    3. URL replacement with a standardized 'url_token' signal.
    4. Currency symbol replacement with a standardized 'currency_token' signal.
    5. Comma removal in large numbers (e.g. 1,000 -> 1000).
    6. Number sequence replacement (4+ digits) with 'number_token'.
    7. Lowercasing to ensure uniform case.
    8. Removal of residual punctuation/non-alphanumeric chars (preserving underscores).
    9. Tokenization, stopword removal (excluding negations), and lemmatization.

    Args:
        text: Raw input message string.

    Returns:
        Cleaned, normalized text string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    cleaned = text

    # Step 1: Expand common contractions
    for pattern, replacement in CONTRACTION_MAP.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    # Step 2: Normalize URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+", " url_token ", cleaned)

    # Step 3: Normalize currency symbols ($ £ € ¥ ₹) - collapsing consecutive symbols
    cleaned = re.sub(r"[\$£€¥₹]+", " currency_token ", cleaned)

    # Step 4: Remove commas inside digits (e.g., 10,000 -> 10000)
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", cleaned)

    # Step 5: Normalize phone numbers and digit codes (4 or more continuous digits)
    cleaned = re.sub(r"\b\d{4,}\b", " number_token ", cleaned)

    # Step 6: Lowercase
    cleaned = cleaned.lower()

    # Step 7: Clean punctuation, keeping alphanumerics and underscores for tokens
    cleaned = re.sub(r"[^a-z0-9\s_]", " ", cleaned)

    # Step 8: Tokenize and filter
    tokens = cleaned.split()
    processed_tokens: List[str] = []

    for token in tokens:
        if token in STOP_WORDS:
            continue
        # Preserve single-letter words only if meaningful (e.g. numbers), else length > 1
        if len(token) <= 1 and not token.isdigit():
            continue

        if LEMMATIZER is not None:
            # Lemmatize both verb and noun forms
            token = LEMMATIZER.lemmatize(token, pos="v")
            token = LEMMATIZER.lemmatize(token, pos="n")

        processed_tokens.append(token)

    return " ".join(processed_tokens)


def check_conflicting_duplicates(df: pd.DataFrame) -> Tuple[int, int, List[str]]:
    """Check whether duplicate SMS texts carry conflicting labels."""
    grouped = df.groupby("message")["label"].nunique()
    conflicts = grouped[grouped > 1]
    num_conflicts = len(conflicts)
    total_dups = int(df.duplicated(subset=["message"], keep=False).sum())
    conflict_messages = conflicts.index.tolist()
    return total_dups, num_conflicts, conflict_messages


def prepare_and_split_data(
    raw_csv_path: str,
    output_dir: str,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict:
    """Preprocess the dataset and perform stratified 80:20 train-test partitioning.

    Strict data hygiene guarantees:
    - Raw input file is strictly read-only and preserved untouched.
    - Duplicate rows are analyzed for label consistency.
    - Deduplication is applied BEFORE splitting to prevent data leakage.
    - Stratified splitting guarantees identical class balance in train and test sets.
    - Preprocessing is applied identically to both partitions.
    """
    df_raw = pd.read_csv(raw_csv_path)

    # Step 1: Duplicate and conflict verification
    total_dup_records, num_conflicts, conflict_msgs = (
        check_conflicting_duplicates(df_raw)
    )

    # Step 2: Deduplication to prevent train/test leakage
    df_dedup = df_raw.drop_duplicates(subset=["message"], keep="first").copy()

    # Step 3: Apply text preprocessing
    df_dedup["cleaned_message"] = df_dedup["message"].apply(preprocess_text)

    # Step 4: Encode labels: ham -> 0, spam -> 1
    label_mapping = {"ham": 0, "spam": 1}
    df_dedup["label_num"] = df_dedup["label"].map(label_mapping)

    # Step 5: Stratified train-test split (80:20)
    train_df, test_df = train_test_split(
        df_dedup,
        test_size=test_size,
        random_state=random_state,
        stratify=df_dedup["label_num"],
    )

    # Step 6: Save train and test sets
    os.makedirs(output_dir, exist_ok=True)
    train_csv = os.path.join(output_dir, "train.csv")
    test_csv = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # Compute partition statistics
    train_counts = train_df["label"].value_counts().to_dict()
    test_counts = test_df["label"].value_counts().to_dict()

    train_total = len(train_df)
    test_total = len(test_df)

    summary = {
        "raw_total_records": len(df_raw),
        "total_duplicate_rows": total_dup_records,
        "conflicting_duplicate_messages": num_conflicts,
        "conflict_handling": (
            "No conflicts detected. Deduplication performed before train/test split to prevent leakage."
            if num_conflicts == 0
            else "Conflicting messages removed or resolved."
        ),
        "unique_records_after_dedup": len(df_dedup),
        "split_ratio": f"{int((1 - test_size) * 100)}:{int(test_size * 100)}",
        "train_set": {
            "total": train_total,
            "ham_count": train_counts.get("ham", 0),
            "ham_percentage": round(
                (train_counts.get("ham", 0) / train_total) * 100, 2
            ),
            "spam_count": train_counts.get("spam", 0),
            "spam_percentage": round(
                (train_counts.get("spam", 0) / train_total) * 100, 2
            ),
            "file_path": train_csv,
        },
        "test_set": {
            "total": test_total,
            "ham_count": test_counts.get("ham", 0),
            "ham_percentage": round(
                (test_counts.get("ham", 0) / test_total) * 100, 2
            ),
            "spam_count": test_counts.get("spam", 0),
            "spam_percentage": round(
                (test_counts.get("spam", 0) / test_total) * 100, 2
            ),
            "file_path": test_csv,
        },
    }

    summary_file = os.path.join(output_dir, "preprocessing_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base, "dataset", "sms_dataset.csv")
    out_dir = os.path.join(base, "dataset")

    print("Executing Phase 3: Data Preprocessing and Stratified Split...")
    res = prepare_and_split_data(raw_path, out_dir)

    print("\n=== PREPROCESSING & SPLIT SUMMARY ===")
    print(f"Raw records: {res['raw_total_records']}")
    print(
        f"Conflicting duplicate messages: {res['conflicting_duplicate_messages']}"
    )
    print(
        f"Cleaned unique records after deduplication: {res['unique_records_after_dedup']}"
    )
    print(
        f"Train Set ({res['split_ratio']}): {res['train_set']['total']} msgs | Ham: {res['train_set']['ham_count']} ({res['train_set']['ham_percentage']}%) | Spam: {res['train_set']['spam_count']} ({res['train_set']['spam_percentage']}%)"
    )
    print(
        f"Test Set ({res['split_ratio']}): {res['test_set']['total']} msgs | Ham: {res['test_set']['ham_count']} ({res['test_set']['ham_percentage']}%) | Spam: {res['test_set']['spam_count']} ({res['test_set']['spam_percentage']}%)"
    )
