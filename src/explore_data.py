"""Dataset loading, conversion, and exploratory data analysis (EDA)."""

import json
import os
import pandas as pd
import numpy as np


def load_and_standardize_dataset(
    raw_path: str, output_csv_path: str
) -> pd.DataFrame:
    """Load raw SMSSpamCollection tab-separated text and save as standard CSV."""
    df = pd.read_csv(
        raw_path,
        sep="\t",
        names=["label", "message"],
        header=None,
        quoting=3,  # QUOTE_NONE to avoid issues with unescaped quotes in SMS
        encoding="utf-8",
    )
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    return df


def analyze_dataset(df: pd.DataFrame) -> dict:
    """Perform thorough inspection of the SMS dataset."""
    total_records = len(df)
    missing_count = int(df.isnull().sum().sum())
    empty_messages = int((df["message"].str.strip() == "").sum())

    # Duplicate statistics
    duplicate_count = int(df.duplicated(subset=["message"]).sum())
    duplicate_pct = round((duplicate_count / total_records) * 100, 2)

    # Class distribution (raw)
    raw_counts = df["label"].value_counts().to_dict()
    raw_pct = {
        k: round((v / total_records) * 100, 2) for k, v in raw_counts.items()
    }

    # Unique messages view
    df_unique = df.drop_duplicates(subset=["message"]).copy()
    unique_total = len(df_unique)
    unique_counts = df_unique["label"].value_counts().to_dict()
    unique_pct = {
        k: round((v / unique_total) * 100, 2) for k, v in unique_counts.items()
    }

    # Text length metrics (raw)
    df["char_length"] = df["message"].str.len()
    df["word_count"] = df["message"].str.split().str.len()

    char_stats = {}
    word_stats = {}
    for lbl in ["ham", "spam"]:
        subset = df[df["label"] == lbl]
        char_stats[lbl] = {
            "min": int(subset["char_length"].min()),
            "max": int(subset["char_length"].max()),
            "mean": round(float(subset["char_length"].mean()), 2),
            "median": round(float(subset["char_length"].median()), 2),
        }
        word_stats[lbl] = {
            "min": int(subset["word_count"].min()),
            "max": int(subset["word_count"].max()),
            "mean": round(float(subset["word_count"].mean()), 2),
            "median": round(float(subset["word_count"].median()), 2),
        }

    unique_labels = sorted(df["label"].unique().tolist())

    summary = {
        "dataset_name": "UCI SMS Spam Collection",
        "dataset_source": "UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/228/sms+spam+collection)",
        "total_records": total_records,
        "unique_records": unique_total,
        "missing_values": missing_count,
        "empty_messages": empty_messages,
        "duplicate_messages": duplicate_count,
        "duplicate_percentage": duplicate_pct,
        "detected_labels": unique_labels,
        "raw_class_distribution": {
            "counts": raw_counts,
            "percentages": raw_pct,
        },
        "deduplicated_class_distribution": {
            "counts": unique_counts,
            "percentages": unique_pct,
        },
        "character_length_stats": char_stats,
        "word_count_stats": word_stats,
        "scam_label_present": "scam" in unique_labels,
    }
    return summary


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, "dataset", "SMSSpamCollection")
    csv_path = os.path.join(base_dir, "dataset", "sms_dataset.csv")
    summary_path = os.path.join(base_dir, "dataset", "dataset_summary.json")

    print(f"Loading raw dataset from: {raw_path}")
    df = load_and_standardize_dataset(raw_path, csv_path)
    print(f"Saved standardized CSV to: {csv_path} ({len(df)} records)")

    print("Analyzing dataset...")
    summary = analyze_dataset(df)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved dataset summary to: {summary_path}")

    print("\n--- DATASET INSPECTION RESULTS ---")
    print(f"Total Rows: {summary['total_records']}")
    print(f"Missing Values: {summary['missing_values']}")
    print(f"Empty Messages: {summary['empty_messages']}")
    print(
        f"Duplicates: {summary['duplicate_messages']} ({summary['duplicate_percentage']}%)"
    )
    print(f"Unique Rows: {summary['unique_records']}")
    print(f"Distinct Labels: {summary['detected_labels']}")
    print(f"Scam Label Present: {summary['scam_label_present']}")
    print(
        f"Raw Class Distribution: Ham = {summary['raw_class_distribution']['counts'].get('ham', 0)} ({summary['raw_class_distribution']['percentages'].get('ham', 0)}%), Spam = {summary['raw_class_distribution']['counts'].get('spam', 0)} ({summary['raw_class_distribution']['percentages'].get('spam', 0)}%)"
    )
    print(
        f"Deduplicated Distribution: Ham = {summary['deduplicated_class_distribution']['counts'].get('ham', 0)} ({summary['deduplicated_class_distribution']['percentages'].get('ham', 0)}%), Spam = {summary['deduplicated_class_distribution']['counts'].get('spam', 0)} ({summary['deduplicated_class_distribution']['percentages'].get('spam', 0)}%)"
    )
    print(f"Character Length Stats: {summary['character_length_stats']}")
    print(f"Word Count Stats: {summary['word_count_stats']}")


if __name__ == "__main__":
    main()
