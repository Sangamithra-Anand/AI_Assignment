"""
sentiment_analysis.py
-----------------------------------------
This file performs SENTIMENT ANALYSIS on the blog posts.

Steps performed:
1. Load cleaned dataset (Cleaned_Text column).
2. Use TextBlob to compute sentiment polarity.
3. Convert polarity into sentiment category:
        > 0  → Positive
        = 0  → Neutral
        < 0  → Negative
4. Add two new columns:
        - Polarity
        - Sentiment
5. Save sentiment results into outputs/sentiment_results.csv
6. Save summary report into reports/sentiment_summary.txt

Everything is explained clearly inside the code.
"""

import os
import pandas as pd
from textblob import TextBlob


# ----------------------------------------------------------------------
# Helper: ensure outputs folder exists
# ----------------------------------------------------------------------
def ensure_outputs_folder(path="outputs/"):
    """
    Creates 'outputs/' folder if missing.
    Used to save sentiment result CSVs.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[AUTO] Created folder: {path}")


# ----------------------------------------------------------------------
# Helper: ensure reports folder exists
# ----------------------------------------------------------------------
def ensure_reports_folder(path="reports/"):
    """
    Creates 'reports/' folder if missing.
    Used to save summary files.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[AUTO] Created folder: {path}")


# ----------------------------------------------------------------------
# Convert polarity score into sentiment label
# ----------------------------------------------------------------------
def get_sentiment_label(polarity):
    """
    Converts polarity value (-1 to +1) into sentiment label.

    polarity > 0   → Positive
    polarity = 0   → Neutral
    polarity < 0   → Negative
    """
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    return "Neutral"


# ----------------------------------------------------------------------
# Main Sentiment Analysis Function
# ----------------------------------------------------------------------
def analyze_sentiment(df):
    """
    Adds sentiment analysis columns to dataset:
        - Polarity score (float)
        - Sentiment label (string)

    Returns:
        DataFrame with sentiment columns.
    """

    print("[INFO] Starting sentiment analysis...")

    # Compute polarity using TextBlob for each blog
    df["Polarity"] = df["Cleaned_Text"].apply(lambda text: TextBlob(text).sentiment.polarity)

    # Convert polarity score into labels
    df["Sentiment"] = df["Polarity"].apply(get_sentiment_label)

    print("[INFO] Sentiment analysis completed.")
    print(df[["Cleaned_Text", "Polarity", "Sentiment"]].head())

    return df


# ----------------------------------------------------------------------
# Save sentiment outputs
# ----------------------------------------------------------------------
def save_sentiment_results(df):
    """
    Saves:
        - Full sentiment results CSV into outputs/
        - Summary statistics into reports/
    """

    ensure_outputs_folder()
    ensure_reports_folder()

    # File paths
    results_path = "outputs/sentiment_results.csv"
    summary_path = "reports/sentiment_summary.txt"

    # Save full sentiment CSV
    df.to_csv(results_path, index=False)
    print(f"[INFO] Sentiment results saved to: {results_path}")

    # Prepare summary
    sentiment_counts = df["Sentiment"].value_counts()

    with open(summary_path, "w") as file:
        file.write("SENTIMENT SUMMARY REPORT\n")
        file.write("------------------------\n")
        for sent, count in sentiment_counts.items():
            file.write(f"{sent}: {count}\n")

    print(f"[INFO] Sentiment summary saved to: {summary_path}")


# ----------------------------------------------------------------------
# Self-Test Block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from feature_engineering import load_cleaned_dataset

    print("[TEST] Running sentiment_analysis.py directly...")

    df_clean = load_cleaned_dataset()
    if df_clean is not None:
        df_with_sentiment = analyze_sentiment(df_clean)
        save_sentiment_results(df_with_sentiment)

    print("[TEST] Sentiment analysis test completed successfully.")
