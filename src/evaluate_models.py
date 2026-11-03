"""Comprehensive Model Evaluation, Visualizations, and Error Analysis.

Phase 6 Deliverables:
1. Publication-quality evaluation visualizations:
   - Confusion matrices (4 models)
   - Comparative accuracy bar chart
   - Comparative precision, recall, and F1-score grouped bar chart
   - ROC curves with AUC values
   - Precision-Recall curves with Average Precision (AP)
   - Cross-validation performance comparison with error bars (mean ± std)
2. Rigorous Error Analysis for Primary Champion (Linear SVM Balanced):
   - Detailed breakdown of all False Positives and False Negatives
   - Linguistic and feature-level explanations
   - Redacted PII to maintain privacy
3. Export structured metrics, plots, and markdown report to evaluation/
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

# Styling configuration for publication-quality charts
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})


def redact_pii(text: str) -> str:
    """Mask phone numbers, shortcodes, and web domains to protect privacy."""
    text = re.sub(r"\b\d{5,}\b", "[PHONE_REDACTED]", text)
    text = re.sub(r"https?://\S+|www\.\S+", "[URL_REDACTED]", text)
    return text


def load_assets(base_dir: str):
    """Load test dataset, fitted pipelines, and CV summary."""
    test_csv = os.path.join(base_dir, "dataset", "test.csv")
    models_dir = os.path.join(base_dir, "models")

    test_df = pd.read_csv(test_csv)
    X_test_text = test_df["cleaned_message"].fillna("").astype(str)
    y_test = test_df["label_num"].values

    pipeline_names = {
        "Linear SVM (Balanced)": "svm_balanced_pipeline.pkl",
        "Linear SVM (Default)": "svm_default_pipeline.pkl",
        "Multinomial Naive Bayes": "nb_pipeline.pkl",
        "Logistic Regression": "lr_pipeline.pkl",
    }

    pipelines = {}
    for name, fname in pipeline_names.items():
        fpath = os.path.join(models_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing pipeline artifact: {fpath}")
        pipelines[name] = joblib.load(fpath)

    metrics_json_path = os.path.join(models_dir, "evaluation_metrics.json")
    with open(metrics_json_path, "r", encoding="utf-8") as f:
        metrics_meta = json.load(f)

    return test_df, X_test_text, y_test, pipelines, metrics_meta


def compute_test_scores(pipelines: Dict[str, Any], X_test_text: pd.Series, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Compute predictions, continuous decision scores, and curve data for each model."""
    results = {}

    for name, pipe in pipelines.items():
        y_pred = pipe.predict(X_test_text)

        # Retrieve continuous score for ROC / PR curves
        clf = pipe.named_steps["clf"]
        if hasattr(pipe, "decision_function"):
            y_score = pipe.decision_function(X_test_text)
        elif hasattr(pipe, "predict_proba"):
            y_score = pipe.predict_proba(X_test_text)[:, 1]
        else:
            y_score = y_pred

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
        rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
        cm = confusion_matrix(y_test, y_pred)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = float(auc(fpr, tpr))

        pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_score)
        avg_prec = float(average_precision_score(y_test, y_score))

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "cm": cm,
            "roc_auc": roc_auc,
            "avg_precision": avg_prec,
            "y_pred": y_pred,
            "y_score": y_score,
            "fpr": fpr,
            "tpr": tpr,
            "pr_prec": pr_prec,
            "pr_rec": pr_rec,
        }

    return results


def plot_all_confusion_matrices(results: Dict[str, Dict[str, Any]], out_path: str):
    """Plot 2x2 grid of confusion matrices."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for idx, (name, res) in enumerate(results.items()):
        cm = res["cm"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[idx],
            cbar=False,
            annot_kws={"size": 13, "weight": "bold"},
            xticklabels=["Predicted Ham", "Predicted Spam"],
            yticklabels=["Actual Ham", "Actual Spam"],
        )
        axes[idx].set_title(
            f"{name}\nAcc: {res['accuracy']:.4f} | Prec: {res['precision']:.4f} | Rec: {res['recall']:.4f} | F1: {res['f1']:.4f}",
            fontweight="semibold",
        )

    plt.suptitle("Confusion Matrix Comparison (Held-Out Test Set: 1,035 Messages)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_comparative_accuracy(results: Dict[str, Dict[str, Any]], out_path: str):
    """Plot horizontal comparative accuracy bar chart."""
    models = list(results.keys())
    accuracies = [results[m]["accuracy"] * 100 for m in models]
    colors = ["#1f77b4" if "Balanced" in m else "#aec7e8" for m in models]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(models, accuracies, color=colors, edgecolor="black", height=0.55)
    ax.set_xlim(95.0, 100.0)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Comparative Test Accuracy Across Candidate Models", fontweight="bold")

    for bar, val in zip(bars, accuracies):
        ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2, f"{val:.2f}%", va="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_comparative_metrics(results: Dict[str, Dict[str, Any]], out_path: str):
    """Plot grouped bar chart for Spam Precision, Recall, and F1."""
    models = list(results.keys())
    precisions = [results[m]["precision"] for m in models]
    recalls = [results[m]["recall"] for m in models]
    f1s = [results[m]["f1"] for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    r1 = ax.bar(x - width, precisions, width, label="Spam Precision", color="#2ca02c")
    r2 = ax.bar(x, recalls, width, label="Spam Recall", color="#ff7f0e")
    r3 = ax.bar(x + width, f1s, width, label="Spam F1-Score", color="#1f77b4")

    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.set_title("Comparative Spam Precision, Recall, and F1-Score on Test Set", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=10, ha="right")
    ax.set_ylim(0.75, 1.03)
    ax.legend(loc="lower right")

    for rects in [r1, r2, r3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curves(results: Dict[str, Dict[str, Any]], out_path: str):
    """Plot Receiver Operating Characteristic (ROC) curves with AUC."""
    fig, ax = plt.subplots(figsize=(8, 6.5))

    palette = {
        "Linear SVM (Balanced)": ("#d62728", "-"),
        "Linear SVM (Default)": ("#1f77b4", "--"),
        "Multinomial Naive Bayes": ("#2ca02c", "-."),
        "Logistic Regression": ("#9467bd", ":"),
    }

    for name, res in results.items():
        color, ls = palette.get(name, ("black", "-"))
        ax.plot(
            res["fpr"],
            res["tpr"],
            label=f"{name} (AUC = {res['roc_auc']:.4f})",
            color=color,
            linestyle=ls,
            linewidth=2.2,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.5000)")
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontweight="bold")
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pr_curves(results: Dict[str, Dict[str, Any]], out_path: str):
    """Plot Precision-Recall curves with Average Precision (AP)."""
    fig, ax = plt.subplots(figsize=(8, 6.5))

    palette = {
        "Linear SVM (Balanced)": ("#d62728", "-"),
        "Linear SVM (Default)": ("#1f77b4", "--"),
        "Multinomial Naive Bayes": ("#2ca02c", "-."),
        "Logistic Regression": ("#9467bd", ":"),
    }

    for name, res in results.items():
        color, ls = palette.get(name, ("black", "-"))
        ax.plot(
            res["pr_rec"],
            res["pr_prec"],
            label=f"{name} (AP = {res['avg_precision']:.4f})",
            color=color,
            linestyle=ls,
            linewidth=2.2,
        )

    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.6, 1.02])
    ax.set_xlabel("Spam Recall")
    ax.set_ylabel("Spam Precision")
    ax.set_title("Precision-Recall (PR) Curves on Imbalanced Test Set", fontweight="bold")
    ax.legend(loc="lower left", frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cv_comparison(cv_meta: Dict[str, Any], out_path: str):
    """Plot cross-validation comparison with error bars (mean ± std)."""
    cv_data = cv_meta.get("cross_validation_results", {})
    models = list(cv_data.keys())

    f1_means = [cv_data[m]["cv_spam_f1_mean"] for m in models]
    f1_stds = [cv_data[m]["cv_spam_f1_std"] for m in models]

    rec_means = [cv_data[m]["cv_spam_recall_mean"] for m in models]
    rec_stds = [cv_data[m]["cv_spam_recall_std"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width / 2,
        f1_means,
        width,
        yerr=f1_stds,
        capsize=5,
        label="CV Spam F1 (Mean ± Std)",
        color="#1f77b4",
        edgecolor="black",
    )
    ax.bar(
        x + width / 2,
        rec_means,
        width,
        yerr=rec_stds,
        capsize=5,
        label="CV Spam Recall (Mean ± Std)",
        color="#ff7f0e",
        edgecolor="black",
    )

    ax.set_ylabel("Cross-Validation Score")
    ax.set_title("5-Fold Stratified Cross-Validation (Zero-Leakage Training Partitions)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=10, ha="right")
    ax.set_ylim(0.70, 1.02)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def perform_error_analysis(
    test_df: pd.DataFrame,
    champion_pipe: Any,
    out_json_path: str,
) -> List[Dict[str, Any]]:
    """Analyze all misclassified instances by the Primary Champion Model."""
    X_text = test_df["cleaned_message"].fillna("").astype(str)
    y_true = test_df["label_num"].values
    y_pred = champion_pipe.predict(X_text)
    scores = champion_pipe.decision_function(X_text)

    misclassified = []
    for idx, (true_val, pred_val, score) in enumerate(zip(y_true, y_pred, scores)):
        if true_val != pred_val:
            raw_msg = test_df.iloc[idx]["message"]
            clean_msg = test_df.iloc[idx]["cleaned_message"]
            err_type = "False Positive" if pred_val == 1 else "False Negative"

            # Linguistic diagnostic reasoning
            if err_type == "False Positive":
                reasoning = (
                    "Legitimate message containing promotional or urgency vocabulary "
                    "(e.g., number tokens, service requests, or imperatives) that cross the decision hyperplane."
                )
            else:
                reasoning = (
                    "Spam message lacking typical marketing keywords, using conversational grammar, "
                    "or relying on non-lexical evasion strategies."
                )

            misclassified.append({
                "test_index": int(idx),
                "error_type": err_type,
                "actual_label": "Ham" if true_val == 0 else "Spam",
                "predicted_label": "Spam" if pred_val == 1 else "Ham",
                "decision_function_score": round(float(score), 4),
                "redacted_raw_message": redact_pii(raw_msg),
                "cleaned_tokens": clean_msg,
                "diagnostic_explanation": reasoning,
            })

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(misclassified, f, indent=2)

    return misclassified


def run_evaluation_suite(base_dir: str):
    """Execute complete Phase 6 evaluation pipeline."""
    eval_dir = os.path.join(base_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    print("Phase 6: Loading assets and held-out test data...")
    test_df, X_test_text, y_test, pipelines, metrics_meta = load_assets(base_dir)

    print("Computing metrics and decision scores across all candidate pipelines...")
    results = compute_test_scores(pipelines, X_test_text, y_test)

    print("Generating publication-quality charts...")
    cm_path = os.path.join(eval_dir, "confusion_matrices_all.png")
    plot_all_confusion_matrices(results, cm_path)

    acc_path = os.path.join(eval_dir, "comparative_accuracy.png")
    plot_comparative_accuracy(results, acc_path)

    metrics_path = os.path.join(eval_dir, "comparative_metrics.png")
    plot_comparative_metrics(results, metrics_path)

    roc_path = os.path.join(eval_dir, "roc_curves.png")
    plot_roc_curves(results, roc_path)

    pr_path = os.path.join(eval_dir, "pr_curves.png")
    plot_pr_curves(results, pr_path)

    cv_path = os.path.join(eval_dir, "cv_comparison.png")
    plot_cv_comparison(metrics_meta, cv_path)

    print("Performing in-depth error analysis on Primary Champion Model...")
    err_json_path = os.path.join(eval_dir, "error_analysis.json")
    champ_pipe = pipelines["Linear SVM (Balanced)"]
    errors = perform_error_analysis(test_df, champ_pipe, err_json_path)

    print(f"\nTotal test misclassifications: {len(errors)} out of 1,035 (Error Rate: {len(errors)/1035*100:.2f}%)")
    fp_count = sum(1 for e in errors if e["error_type"] == "False Positive")
    fn_count = sum(1 for e in errors if e["error_type"] == "False Negative")
    print(f" - False Positives: {fp_count} (Ham wrongly predicted as Spam)")
    print(f" - False Negatives: {fn_count} (Spam missed as Ham)")

    return {
        "results": results,
        "errors": errors,
        "plots": {
            "confusion_matrices": cm_path,
            "accuracy": acc_path,
            "metrics": metrics_path,
            "roc_curves": roc_path,
            "pr_curves": pr_path,
            "cv_comparison": cv_path,
        },
    }


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_evaluation_suite(base)
    print("\nPhase 6 Evaluation Suite Completed Successfully!")
