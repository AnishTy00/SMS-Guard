# Dataset Documentation & Exploration Report

## 1. Dataset Source
- **Origin**: UCI Machine Learning Repository — SMS Spam Collection Dataset
- **URL**: [https://archive.ics.uci.edu/dataset/228/sms+spam+collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Direct Download**: `https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip`
- **Citation**: Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering: New Collection and Results. In *Proceedings of the 2011 ACM Symposium on Document Engineering (DocEng'11)*, Mountain View, CA, USA.

---

## 2. Dataset Files in `dataset/`
- `SMSSpamCollection`: Original tab-separated raw text file from UCI (`label \t message`).
- `sms_dataset.csv`: Standardized 2-column CSV (`label`, `message`) preserved read-only.
- `train.csv`: Stratified 80% training partition (4,136 messages).
- `test.csv`: Stratified 20% held-out testing partition (1,035 messages).
- `dataset_summary.json`: Machine-readable summary statistics.
- `preprocessing_summary.json`: Partitioning metrics and duplicate analysis.

---

## 3. Data Quality & Duplicate Breakdown

### Clarification on Duplicate Metrics
To ensure full academic rigor, duplicate metrics are precisely defined as follows:

| Metric | Count | Description |
| :--- | :--- | :--- |
| **Raw Records** | 5,574 | Total SMS rows in the original UCI dataset. |
| **Distinct Duplicate Groups** | 281 | Unique message texts that appear two or more times in the dataset. |
| **Total Occurrences in Duplicate Groups** | 684 | Total rows comprising the 281 duplicated messages (`keep=False`). |
| **Excess Redundant Rows Removed** | 403 | The extra duplicate copies dropped during deduplication ($684 - 281 = 403$). |
| **Conflicting Labels Among Duplicates** | 0 | Every duplicate message possesses 100% consistent ground-truth labels across all its copies. |
| **Final Unique Records After Deduplication** | 5,171 | Distinct messages remaining for splitting ($5,574 - 403 = 5,171$). |

---

## 4. Class Distribution Comparison

| Partition | Total Messages | Ham Count (%) | Spam Count (%) | Class Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **Raw Dataset** | 5,574 | 4,827 (86.60%) | 747 (13.40%) | ~6.5 : 1 |
| **Deduplicated Dataset** | 5,171 | 4,518 (87.37%) | 653 (12.63%) | ~6.9 : 1 |
| **Train Set (80%)** | 4,136 | 3,614 (87.38%) | 522 (12.62%) | ~6.9 : 1 |
| **Test Set (20%)** | 1,035 | 904 (87.34%) | 131 (12.66%) | ~6.9 : 1 |

*Stratified 80:20 partitioning ensures identical class balance in train and test sets, differing by <0.04%.*

---

## 5. Text Length and Word Count Analysis

| Category | Character Length (Mean / Median / Min / Max) | Word Count (Mean / Median / Min / Max) |
| :--- | :--- | :--- |
| **Ham (Legitimate)** | 71.47 / 52.0 / 2 / 910 | 14.30 / 11.0 / 1 / 171 |
| **Spam** | 138.68 / 149.0 / 13 / 223 | 23.91 / 25.0 / 2 / 35 |
