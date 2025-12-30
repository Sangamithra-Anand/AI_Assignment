"""
compare_results.py
---------------------------------------------------------
This file compares the clustering results between:

1. K-Means on ORIGINAL SCALED DATA
2. K-Means on PCA-TRANSFORMED DATA

WHAT THIS FILE DOES:
---------------------
1. Loads both clustering metrics from JSON files
2. Compares:
      ✔ Silhouette Scores
      ✔ Davies–Bouldin Index
3. Determines which approach performed better
4. Saves a readable comparison report for final analysis

UTF-8 FIX:
----------
Windows cannot save emojis using default cp1252 encoding.
We now use encoding="utf-8" when writing the report to avoid
charmap encoding errors.
"""

import os
import json


REPORTS_DIR = "outputs/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)



# ========================================================================
# FUNCTION: load_scores(json_path)
# PURPOSE :
#   Loads clustering evaluation metrics from a JSON file.
# ========================================================================
def load_scores(json_path):
    """
    Loads metrics from a JSON file.

    Args:
        json_path (str): Path to the metrics JSON file

    Returns:
        dict: Loaded JSON data
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"[ERROR] File not found: {json_path}")

    with open(json_path, "r") as f:
        return json.load(f)



# ========================================================================
# FUNCTION: compare_clustering_results()
# PURPOSE :
#   Compares evaluation metrics from original and PCA-based clustering.
# ========================================================================
def compare_clustering_results():
    """
    Compares clustering results from original scaled data
    and PCA-transformed data.

    Saves a markdown report summarizing:
        ✔ Silhouette comparison
        ✔ DB Index comparison
        ✔ Which method is better and why
    """

    print("\n[INFO] Comparing clustering results...")

    # Load evaluation scores for both methods
    original_path = f"{REPORTS_DIR}/clustering_original_scores.json"
    pca_path = f"{REPORTS_DIR}/clustering_pca_scores.json"

    original_scores = load_scores(original_path)
    pca_scores = load_scores(pca_path)

    sil_original = original_scores["silhouette_score"]
    sil_pca = pca_scores["silhouette_score"]

    db_original = original_scores["davies_bouldin_index"]
    db_pca = pca_scores["davies_bouldin_index"]

    n_clusters = original_scores["n_clusters"]


    # Decide which model is better for each metric
    better_silhouette = "PCA" if sil_pca > sil_original else "Original"
    better_db = "PCA" if db_pca < db_original else "Original"

    verdict = (
        "PCA-based Clustering"
        if better_silhouette == "PCA" and better_db == "PCA"
        else "Original Scaled Data Clustering"
    )


    # Build markdown report
    report_path = f"{REPORTS_DIR}/comparison_report.md"

    report_text = f"""
# Clustering Comparison Report
Number of Clusters: {n_clusters}

---

## 📊 Silhouette Score Comparison
| Method                   | Score     |
|--------------------------|-----------|
| Original Scaled Data     | {sil_original:.4f} |
| PCA-Transformed Data     | {sil_pca:.4f} |

**Better Silhouette Score:** {better_silhouette}

---

## 📉 Davies–Bouldin Index Comparison
| Method                   | DB Index  |
|--------------------------|-----------|
| Original Scaled Data     | {db_original:.4f} |
| PCA-Transformed Data     | {db_pca:.4f} |

**Better DB Index:** {better_db}

---

## 🏆 Final Verdict
**Best Overall Clustering Method:**  
### 👉 {verdict}

---

## 📝 Notes
- PCA may improve clustering by reducing noise and removing correlated features.
- But if PCA removes important structure, original data may perform better.
"""


    # Save report with UTF-8 encoding (Windows FIX)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"[INFO] Comparison report saved at: {report_path}")
    print("\n[INFO] Comparison completed successfully!\n")



# ========================================================================
# TESTING BLOCK — runs only when executed directly
# ========================================================================
if __name__ == "__main__":
    print("[TEST] Running compare_results.py directly...")

    try:
        compare_clustering_results()
        print("[TEST] compare_results.py executed successfully.")
    except Exception as e:
        print("[TEST ERROR]", e)

