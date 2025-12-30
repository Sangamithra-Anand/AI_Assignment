"""
main.py
-------------------------
Groceries-Only Pipeline
-------------------------
This version ensures ALL reports, visuals, and logs are saved INSIDE the output folder.

Folders created:
- data/processed
- output
    - output/reports
    - output/visuals
    - output/logs

Pipeline:
1. Load data
2. Preprocess groceries dataset   (saves cleaned_groceries.csv)
3. Create basket format           (saves to output/)
4. Run Apriori
5. Generate rules
6. Analyze rules (saves visuals + reports)
"""

from utils import (
    ensure_folders_exist,
    start_timer,
    end_timer,
    log_message,
    summarize_dataset,
)

import pandas as pd

from apriori_model import run_apriori
from generate_rules import generate_rules
from analyze_rules import analyze_rules
from load_data import load_raw_data


# ---------------------------------------------------------
# PREPROCESS GROCERIES DATASET  (UPDATED)
# ---------------------------------------------------------
def preprocess_groceries(df):
    """
    Preprocesses groceries dataset.
    Each row is a string of comma-separated items.
    ALSO saves cleaned dataset to:
        data/processed/cleaned_groceries.csv
    """

    print("[INFO] Preprocessing Groceries dataset...")

    # Convert the single long text column into a list of items
    df["items"] = df.iloc[:, 0].astype(str).apply(
        lambda x: [i.strip() for i in x.split(",")]
    )

    print("[INFO] Groceries preprocessing complete.")

    # SAVE CLEANED DATASET
    output_path = "data/processed/cleaned_groceries.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[INFO] Cleaned groceries dataset saved to: {output_path}")

    return df


# ---------------------------------------------------------
# CREATE BASKET FORMAT (UPDATED PATH)
# ---------------------------------------------------------
def create_groceries_basket(df):
    """
    Convert list of items into one-hot encoded basket format.
    Saved inside output/ folder.
    """

    print("[INFO] Creating groceries basket format...")

    basket = df["items"].str.join("|").str.get_dummies(sep="|")

    basket_path = "output/basket_format.csv"
    basket.to_csv(basket_path, index=False)

    print(f"[INFO] Basket saved at {basket_path}")
    print(f"[INFO] Basket shape: {basket.shape}")

    return basket


# ---------------------------------------------------------
# SETUP FOLDERS (UPDATED)
# ---------------------------------------------------------
def setup_project():
    folders = [
        "data/processed",
        "output",
        "output/reports",
        "output/visuals",
        "output/logs",
    ]
    ensure_folders_exist(folders)


# ---------------------------------------------------------
# MAIN PIPELINE (GROCERIES ONLY)
# ---------------------------------------------------------
def run_groceries_pipeline():

    log_message("[PIPELINE] Starting Groceries Dataset Pipeline",
                log_file="output/logs/run_log.txt")

    # Step 1: Load dataset
    t1 = start_timer()
    df_raw = load_raw_data()
    end_timer(t1, "Loading dataset")

    # Step 2: Preprocess
    t2 = start_timer()
    df_clean = preprocess_groceries(df_raw)
    end_timer(t2, "Preprocessing groceries dataset")

    summarize_dataset(df_clean)

    # Step 3: Create basket format
    t3 = start_timer()
    basket = create_groceries_basket(df_clean)
    end_timer(t3, "Creating basket format")

    # Step 4: Apriori
    t4 = start_timer()
    frequent_itemsets = run_apriori(basket, min_support=0.01)
    end_timer(t4, "Running Apriori")

    # Step 5: Generate rules
    t5 = start_timer()
    rules = generate_rules(frequent_itemsets, min_confidence=0.2)
    end_timer(t5, "Generating rules")

    # Step 6: Analyze rules
    t6 = start_timer()
    summary = analyze_rules(rules)  # analyze_rules now writes reports inside output/
    end_timer(t6, "Analyzing rules")

    # Final summary
    print("\n===== PIPELINE COMPLETE =====")
    print(f"Total Rules: {summary['total_rules']}")
    print("=============================\n")


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    setup_project()
    run_groceries_pipeline()


