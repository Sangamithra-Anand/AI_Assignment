"""
apriori_model.py
------------------------
This file performs two major tasks:

1. Convert the cleaned dataset into "Basket Format" 
   required for Association Rule Mining (one-hot encoding).

2. Run the Apriori algorithm to generate Frequent Itemsets.

OUTPUT FILES CREATED:
- data/transactions/basket_format.csv
- output/frequent_itemsets.csv

This file is manually created by you, but all output files
are auto-generated.
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori


def create_basket_format(df):
    """
    Converts the cleaned dataset into a basket (one-hot encoded) format.

    Required for Apriori algorithm.

    Example format:

    InvoiceNo | productA | productB | productC | ...
    ----------------------------------------------
      10001   |     1    |     0    |    1     |
      10002   |     0    |     1    |    1     |

    Args:
        df (DataFrame): Cleaned dataset

    Returns:
        DataFrame: Basket (one-hot encoded transaction matrix)
    """

    print("[INFO] Creating basket (one-hot encoded) format...")

    # ----------------------------------------------------
    # STEP 1: Group by InvoiceNo and Description
    # ----------------------------------------------------
    # We count how many of each item appear in each invoice.
    basket = (
        df.groupby(["InvoiceNo", "Description"])["Quantity"]
        .sum()
        .unstack(fill_value=0)
    )

    # ----------------------------------------------------
    # STEP 2: Convert quantities into one-hot encoding (0/1)
    # ----------------------------------------------------
    # If quantity > 0 → item is present in the transaction.
    print("[INFO] Converting quantities into binary values (0 or 1)...")
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    # ----------------------------------------------------
    # STEP 3: Save basket format for reference
    # ----------------------------------------------------
    output_path = "data/transactions/basket_format.csv"
    basket.to_csv(output_path)
    print(f"[INFO] Basket format saved to: {output_path}")
    print(f"[INFO] Basket matrix shape: {basket.shape}")

    return basket


def run_apriori(basket, min_support=0.02):
    """
    Runs the Apriori algorithm to generate frequent itemsets.

    Args:
        basket (DataFrame): One-hot encoded transaction matrix.
        min_support (float): Minimum support threshold.

    Returns:
        DataFrame: Frequent itemsets with their support values.
    """

    print("[INFO] Running Apriori algorithm...")
    print(f"[INFO] Minimum support threshold = {min_support}")

    # ----------------------------------------------------
    # STEP 1: Generate frequent itemsets using Apriori
    # ----------------------------------------------------
    # use_colnames=True → shows product names instead of column indexes.
    # Low min_support = finds more rules (slow)
    # High min_support = fewer rules (fast)
    frequent_itemsets = apriori(
        basket,
        min_support=min_support,
        use_colnames=True
    )

    print("[INFO] Apriori completed.")
    print(f"[INFO] Total frequent itemsets found: {len(frequent_itemsets)}")

    # ----------------------------------------------------
    # STEP 2: Sort by support (highest first)
    # ----------------------------------------------------
    frequent_itemsets = frequent_itemsets.sort_values(
        by="support", ascending=False
    )

    # ----------------------------------------------------
    # STEP 3: Save frequent itemsets to output folder
    # ----------------------------------------------------
    output_path = "output/frequent_itemsets.csv"
    frequent_itemsets.to_csv(output_path, index=False)

    print(f"[INFO] Frequent itemsets saved to: {output_path}")

    return frequent_itemsets


# ------------------------------------------------------------
# TEST MODE (Runs when executing this file directly)
# ------------------------------------------------------------
if __name__ == "__main__":
    print("[TEST] Running apriori_model.py directly...")

    from load_data import load_raw_data
    from preprocess import preprocess_data

    # Load + preprocess
    df_raw = load_raw_data()
    df_clean = preprocess_data(df_raw)

    # Create basket format
    basket_df = create_basket_format(df_clean)

    # Run Apriori
    frequent_sets = run_apriori(basket_df, min_support=0.02)

    print("[TEST] Frequent itemsets (top 5):")
    print(frequent_sets.head())


