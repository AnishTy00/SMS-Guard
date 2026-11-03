# SMS GUARD: Intelligent SMS Spam and Scam Detection System

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5.2-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0-red.svg)](https://streamlit.io/)
[![License: Academic](https://img.shields.io/badge/License-Academic-burgundy.svg)](#)

An academic AIML project implementing end-to-end Natural Language Processing (NLP) and Supervised Machine Learning for real-time detection of SMS spam, phishing, and scam messages.

---

## 1. Project Overview

Short Message Service (SMS) remains one of the most ubiquitous, trusted, and immediate communication channels globally, commanding open rates exceeding 98%. However, this trustworthiness has made mobile messaging a prime target for smishing (SMS phishing), fraudulent financial solicitations, credential-harvesting schemes, and bulk unsolicited commercial spam.

**SMS GUARD** is a complete, production-grade text classification and security analysis application. Designed in strict compliance with the academic curriculum (**Module 4: Supervised Learning** and **Module 5: Support Vector Machines**), SMS GUARD couples a soft-margin Linear Support Vector Classifier with an explainable, deterministic Heuristic Risk Engine. The system achieves a **99.23% Test Accuracy** and **95.42% Spam Recall** on quarantined benchmark data while operating strictly in memory without persistent user data storage.

---

## 2. Key Features

- **Primary Statistical Classifier (Module 5 Champion)**:
  - Powered by `LinearSVC` with cost-sensitive inverse class frequency weighting (`class_weight='balanced'`).
  - Maximizes geometric separation margin ($\frac{2}{\|\mathbf{w}\|}$) in a 4,000-dimensional TF-IDF feature space.
  - Generates continuous signed decision distances ($f(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + b$) and monotonic relative confidence scores.
- **Strict Methodological Hygiene (Module 4 Alignment)**:
  - **Zero Data Leakage**: Full text deduplication (5,574 to 5,171 messages) and physically quarantined 80:20 stratified splitting (4,136 train, 1,035 test).
  - **Unbiased Model Selection**: 5-Fold Stratified Cross-Validation conducted strictly within atomic `Pipeline` objects on training folds; the held-out test set was never accessed during model selection or tuning.
- **Transparent Heuristic Risk Engine**:
  - Independent rule layer scanning for high-risk threat indicators: shortened URLs, raw IPv4 address links, OTP and credential harvesting, coercive urgency phrasing, and premium SMS shortcodes.
  - Yields transparent, explainable evidence strings and an advisory risk score (0–100) without interfering with the ML decision boundary.
- **Interactive Web Dashboard (Streamlit)**:
  - Custom institutional burgundy user interface (`#8b1e30`).
  - **Tab 1 (SMS Message Analyzer)**: Real-time text analysis, rapid-test presets (Personal Ham, Bank Phishing, Lottery Scam, Negation), prediction badges, decision distances, and risk audits.
  - **Tab 2 (Model Performance & Benchmarks)**: Live KPI metric cards dynamically loaded from `evaluation_metrics.json`, benchmark comparison tables, and high-resolution chart galleries.
  - **Tab 3 (Academic Foundations & Architecture)**: Theoretical syllabus walkthroughs and end-to-end system dataflow architecture.
- **Robust Security & Privacy**:
  - Zero disk or database persistence of analyzed text.
  - Full HTML and XSS escaping (`html.escape()`) on all rendered user text and evidence strings.

---

## 3. Technology Stack

| Layer / Component | Technology | Version | Description |
| :--- | :--- | :--- | :--- |
| **Programming Language** | Python | 3.13.0 | Core development language (64-bit) |
| **Machine Learning** | Scikit-learn | 1.5.2 | TF-IDF vectorization, Pipeline, LinearSVC, Naive Bayes, Logistic Regression |
| **Web Application** | Streamlit | 1.39.0 | Interactive web dashboard and reactive UI components |
| **Data Processing** | Pandas | 2.2.3 | Dataset exploration, deduplication, and stratified partitioning |
| **Numerical Computing** | NumPy | 2.1.2 | High-performance array operations and matrix evaluations |
| **Model Serialization** | Joblib | 1.4.2 | Atomic pipeline persistence and artifact caching |
| **Visualizations** | Matplotlib / Seaborn | 3.9.2 / 0.13.2 | Publication-grade ROC, PR, CV, and confusion matrix charts |
| **Document Generation** | python-docx / ReportLab | 1.2.0 / 5.1.0 | Academic project report compilation (DOCX and PDF formats) |

---

## 4. Academic Syllabus Alignment

### A. Module 5: Support Vector Machine (SVM) — Primary Classifier
- **Model Architecture**: `LinearSVC(C=1.0, class_weight='balanced', random_state=42)`
- **Mathematical Principle**:
  Each SMS message is represented as a high-dimensional TF-IDF vector $\mathbf{x}_i \in \mathbb{R}^D$ ($D = 4000$) with binary class $y_i \in \{-1, +1\}$ ($\text{Ham} = -1, \text{Spam} = +1$). The linear SVM searches for the optimal separating hyperplane:
  $$\mathbf{w}^T \mathbf{x} + b = 0$$
  which maximizes the geometric separation margin $\frac{2}{\|\mathbf{w}\|}$ while penalizing soft-margin slack variables $\xi_i \ge 0$:
  $$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N w_{y_i} \xi_i \quad \text{s.t.} \quad y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
- **Class Weight Balancing**:
  In imbalanced text streams ($3,614$ Ham vs $522$ Spam in training), standard unweighted SVM shifts the boundary toward minority spam. Setting `class_weight='balanced'` assigns inverse frequency weights:
  $$w_c = \frac{N}{2 \times N_c} \implies w_{\text{spam}} \approx 3.96, \quad w_{\text{ham}} \approx 0.57$$
  Penalizing spam errors $\approx 6.92\times$ more heavily widens the spam safety margin, boosting Spam Recall from 92.37% to 95.42% while retaining 98.43% precision.
- **Continuous Decision Score vs. Calibration Semantics**:
  Continuous predictions are determined by the perpendicular signed hyperplane distance:
  $$f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$$
  In `app.py`, this score is transformed via the standard logistic sigmoid:
  $$\text{Relative Confidence Estimate} = \sigma(f(\mathbf{x})) = \frac{1}{1 + e^{-f(\mathbf{x})}}$$
  *Important Note on Calibration*: Because two-parameter Platt scaling (`CalibratedClassifierCV`) was not fitted on an independent split, this percentage represents an **uncalibrated relative confidence estimate** (monotonic squashing of the margin distance), not a statistically calibrated posterior probability.

### B. Module 4: Supervised Learning & Unbiased Methodology
1. **Model Selection via Zero-Leakage 5-Fold Stratified Cross-Validation on Training Text**:
   Evaluated strictly within `train.csv` (4,136 samples) using atomic Scikit-learn `Pipeline` objects. The `TfidfVectorizer` was fitted **independently inside each fold's training split**, completely preventing vocabulary and IDF statistics from leaking into validation folds.
2. **Unseen Generalization Assessment on Held-Out Test Data**:
   Evaluated on `test.csv` (1,035 samples: 904 Ham, 131 Spam) strictly after champion selection was finalized.

---

## 5. Model Architecture & Pipeline Dataflow

```
Incoming SMS Text
       │
       ▼
[Text Preprocessing]
  ├── Case folding to lowercase
  ├── Semantic tokenization: URLs -> 'url_token', Currencies -> 'currency_token', Digits -> 'number_token'
  ├── Negation preservation: 'not', 'no', 'never', 'nor', 'without' strictly retained
  └── Punctuation and non-alphanumeric cleaning
       │
       ├───────────────────────────────────────────────────┐
       ▼                                                   ▼
[Atomic Scikit-learn Pipeline]                     [Heuristic Risk Engine]
  ├── Sublinear TF-IDF (1-2 N-Grams, D=4000)         ├── Regex Scanner (URLs, IPs, OTPs, Urgency, Shortcodes)
  └── LinearSVC (Balanced, C=1.0)                    └── Additive Risk Score (0-100) & Evidence Extraction
       │                                                   │
       ▼                                                   ▼
Hyperplane Decision f(x) = w^T x + b                 Advisory Risk Level & Security Flags
  ├── Prediction: SPAM (f>0) / HAM (f<=0)                  │
  └── Sigmoid Squashing: Relative Confidence %             │
       │                                                   │
       └─────────────────────────┬─────────────────────────┘
                                 ▼
                 [Interactive Streamlit Dashboard]
                    • Tab 1: SMS Message Analyzer
                    • Tab 2: Model Performance & Benchmarks
                    • Tab 3: Academic Foundations & Architecture
```

---

## 6. Empirical Evaluation & Benchmarks

### A. Zero-Leakage 5-Fold Stratified Cross-Validation (4,136 Training Messages)

| Model | Academic Role | CV Accuracy | CV Spam Precision | CV Spam Recall | CV Spam F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear SVM (Balanced)** | **Champion (Module 5)** | **0.9877 ± 0.0039** | **0.9631 ± 0.0225** | **0.9387 ± 0.0179** | **0.9505 ± 0.0155** |
| **Linear SVM (Default)** | Unweighted Baseline | 0.9869 ± 0.0055 | 0.9793 ± 0.0150 | 0.9156 ± 0.0342 | 0.9462 ± 0.0235 |
| **Multinomial Naive Bayes** | Probabilistic Baseline | 0.9763 ± 0.0055 | 0.9975 ± 0.0050 | 0.8141 ± 0.0418 | 0.8960 ± 0.0264 |
| **Logistic Regression** | Linear Baseline | 0.9732 ± 0.0061 | 0.9902 ± 0.0089 | 0.7949 ± 0.0455 | 0.8812 ± 0.0305 |

### B. Held-Out Quarantined Test Evaluation (1,035 Messages: 904 Ham, 131 Spam)

| Model | Test Accuracy | Spam Precision | Spam Recall | Spam F1 | ROC AUC | Average Precision | FP | FN | Train Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear SVM (Balanced)** ★ | **0.9923** | **0.9843** | **0.9542** | **0.9690** | **0.9984** | **0.9918** | **2** | **6** | **0.1777s** |
| **Linear SVM (Default)** | 0.9903 | 1.0000 | 0.9237 | 0.9603 | 0.9990 | 0.9942 | 0 | 10 | 0.1646s |
| **Multinomial Naive Bayes** | 0.9778 | 0.9909 | 0.8321 | 0.9046 | 0.9911 | 0.9732 | 1 | 22 | 0.1546s |
| **Logistic Regression** | 0.9826 | 1.0000 | 0.8626 | 0.9262 | 0.9972 | 0.9869 | 0 | 18 | 0.1926s |

### C. Confusion Matrix Decomposition (Held-Out Test Set: 1,035 Messages)

```text
Linear SVM (Balanced) — Primary Champion:
┌──────────────┬──────────────────┬──────────────────┐
│              │ Predicted Ham    │ Predicted Spam   │
├──────────────┼──────────────────┼──────────────────┤
│ Actual Ham   │ 902 (TN)         │   2 (FP)         │
│ Actual Spam  │   6 (FN)         │ 125 (TP)         │
└──────────────┴──────────────────┴──────────────────┘
• Total Misclassifications: 8 out of 1,035 instances (0.77% error rate).
• False Positives (2): Ultra-short messages ("send content page", "not respond") dominated by marketing imperative keywords.
• False Negatives (6): Slang-heavy brand promos, adult chat, or pop-culture trivia disguised as conversational messages.
```

---

## 7. Project Directory Structure

```text
SMSANDSPAMDETECTION/
├── .streamlit/
│   └── config.toml                 # Burgundy theme styling (#8b1e30)
├── .venv/                          # Isolated Python virtual environment
├── dataset/
│   ├── SMSSpamCollection           # Raw UCI benchmark corpus (5,574 rows)
│   ├── sms_dataset.csv             # Standardized CSV
│   ├── train.csv                   # Stratified train partition (4,136 rows)
│   ├── test.csv                    # Stratified quarantined test set (1,035 rows)
│   ├── dataset_summary.json        # EDA telemetry & length distributions
│   ├── preprocessing_summary.json  # Preprocessing statistics
│   └── README.md                   # Dataset origin documentation
├── docs/
│   ├── SMS_GUARD_Final_Academic_Project_Report.docx  # Comprehensive academic report (DOCX)
│   └── SMS_GUARD_Final_Academic_Project_Report.pdf   # Matching academic report (PDF)
├── evaluation/
│   ├── system_architecture.png     # End-to-end dataflow architecture diagram
│   ├── cv_comparison.png           # 5-fold CV comparison chart with error bars
│   ├── comparative_accuracy.png    # Model accuracy comparison bar chart
│   ├── comparative_metrics.png     # Precision, Recall, and F1 grouped bar chart
│   ├── confusion_matrices_all.png  # 4-model confusion matrix heatmaps
│   ├── roc_curves.png              # Receiver Operating Characteristic curves (AUC=0.9984)
│   ├── pr_curves.png               # Precision-Recall curves (AP=0.9918)
│   ├── screenshot_tab1_verified.png# Tab 1 verified application screenshot
│   ├── screenshot_tab2_verified.png# Tab 2 verified dashboard screenshot
│   ├── screenshot_tab3_verified.png# Tab 3 verified academic view screenshot
│   ├── error_analysis.json         # Deep diagnostic breakdown of 8 misclassifications
│   ├── evaluation_report.md        # Comprehensive Phase 6 academic report
│   └── phase8_integration_report.md# Full Phase 8 integration & security report
├── models/
│   ├── spam_classifier.pkl         # Champion production pipeline (Balanced LinearSVC)
│   ├── pipeline_champion.pkl       # Champion pipeline alias
│   ├── svm_balanced_pipeline.pkl   # Balanced Linear SVM full pipeline
│   ├── svm_default_pipeline.pkl    # Default Linear SVM full pipeline
│   ├── nb_pipeline.pkl             # Multinomial Naive Bayes pipeline
│   ├── lr_pipeline.pkl             # Logistic Regression pipeline
│   ├── tfidf_vectorizer.pkl        # Standalone TF-IDF vectorizer artifact
│   ├── tfidf_summary.json          # Vocabulary metadata (4,000 features)
│   └── evaluation_metrics.json     # Dynamic JSON metrics backing Tab 2 dashboard
├── src/
│   ├── __init__.py
│   ├── explore_data.py             # EDA & deduplication routine
│   ├── preprocessing.py            # Text normalization & negation preservation
│   ├── feature_extraction.py       # TF-IDF pipeline builder & serializer
│   ├── train_model.py              # Supervised model training & CV evaluation
│   ├── evaluate_models.py          # Publication-grade chart generation
│   ├── risk_engine.py              # Deterministic heuristic pattern scanner
│   └── prediction.py               # Inference wrapper with decision score computation
├── tests/
│   ├── test_preprocessing.py       # Preprocessing unit & data leakage tests
│   ├── test_feature_extraction.py  # TF-IDF matrix & pipeline tests
│   ├── test_model_training.py      # Champion model & artifact loading tests
│   └── test_integration_phase8.py  # 13-category end-to-end integration & security suite
├── app.py                          # Interactive Streamlit web application
├── requirements.txt                # Pinned production dependencies
└── README.md                       # Master project documentation
```

---

## 8. Installation & Setup Instructions

### Prerequisites
- Operating System: Windows 10/11, Linux (Ubuntu/Debian), or macOS
- Python Runtime: Python 3.10 to 3.13 (64-bit)

### Step 1: Clone or Navigate to the Repository
```bash
cd "C:\Users\anish\Downloads\SMSANDSPAMDETECTION"
```

### Step 2: Create and Activate Virtual Environment
```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

---

## 9. Application Launch & Execution Commands

### Launch Interactive Streamlit Application
```powershell
& ".\.venv\Scripts\streamlit.exe" run "app.py"
```
*The web interface will automatically launch at `http://localhost:8501`.*

### Execute Full Test Suites
```powershell
# Run Unit & Preprocessing Tests
& ".\.venv\Scripts\python.exe" "tests/test_preprocessing.py"

# Run Feature Extraction Tests
& ".\.venv\Scripts\python.exe" "tests/test_feature_extraction.py"

# Run Model Training & Pipeline Tests
& ".\.venv\Scripts\python.exe" "tests/test_model_training.py"

# Run Comprehensive Phase 8 Integration & Security Suite (13 Tests)
& ".\.venv\Scripts\python.exe" -m unittest "tests/test_integration_phase8.py" -v
```

### Re-run Model Training & Evaluation Pipelines
```powershell
# Run Supervised Model Training & Cross-Validation
& ".\.venv\Scripts\python.exe" "src/train_model.py"

# Re-generate Evaluation Charts & Error Analysis
& ".\.venv\Scripts\python.exe" "src/evaluate_models.py"
```

---

## 10. Scope Boundaries & Limitations

1. **Modality Constraint**: The system is tailored for text-based Short Message Service (SMS) messages and does not inspect Multimedia Messaging Service (MMS) video, audio, or embedded image attachments.
2. **Language Coverage**: The vocabulary and TF-IDF representations are optimized for English-language telecommunication. Multi-lingual messages in regional scripts require language-specific subword tokenizers.
3. **Typographical Obfuscation**: Adversaries utilizing complex homoglyph substitutions (e.g., Cyrillic characters mimicking Latin letters) or zero-day slang absent from the training vocabulary may evade statistical classification unless detected by the heuristic layer.
4. **Decision Score Calibration**: The reported percentage represents a monotonic sigmoid transformation of the SVM decision score (relative confidence estimate), not a statistically calibrated posterior probability distribution.

---

## 11. Final Academic Submission Deliverables

As part of **Phase 9: Final Project Documentation and Submission Preparation**, complete academic reports have been compiled and archived in the `docs/` directory:
- **Academic Project Report (DOCX)**: `docs/SMS_GUARD_Final_Academic_Project_Report.docx` (1.70 MB)
- **Academic Project Report (PDF)**: `docs/SMS_GUARD_Final_Academic_Project_Report.pdf` (2.15 MB)

Both reports include complete front matter (Bonafide Certificate, Candidate Declaration, Abstract, Table of Contents), institutional placeholders (`[Student Name]`, `[Guide Name]`, `[College / Institution Name]`), 10 full chapters, mathematical formulations, 10 embedded high-resolution figures, diagnostic error analyses, and authentic IEEE/APA scholarly citations.

---

## 12. References

1. **Almeida, T. A., Hidalgo, J. M. G., & Yamakami, A. (2011)**. Contributions to the study of SMS spam filtering: New collection and results. *Proceedings of the 2011 ACM Symposium on Document Engineering (DocEng '11)*, pp. 259–262.
2. **Cortes, C., & Vapnik, V. (1995)**. Support-vector networks. *Machine Learning*, 20(3), pp. 273–297.
3. **Joachims, T. (1998)**. Text categorization with Support Vector Machines: Learning with many relevant features. *European Conference on Machine Learning (ECML '98)*, LNCS 1398, pp. 137–142.
4. **Salton, G., & Buckley, C. (1988)**. Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), pp. 513–523.
5. **Pedregosa, F., et al. (2011)**. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, pp. 2825–2830.
6. **Platt, J. C. (1999)**. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*, 10(3), pp. 61–74.
