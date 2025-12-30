"""
preprocess.py
------------------------
This file is responsible for CLEANING the text data.

Steps performed:
1. Handle missing values.
2. Convert text to lowercase.
3. Remove punctuation.
4. Remove numbers.
5. Remove stopwords.
6. Tokenize words.
7. Join the cleaned tokens back into text.
8. Save processed dataset into data/processed/

This file does NOT handle TF-IDF. That will be done in feature_engineering.py.
"""

import os
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ----------------------------------------------------------------------
# Helper: Ensure processed folder exists
# ----------------------------------------------------------------------
def ensure_processed_folder(path="data/processed/"):
    """
    Creates the processed folder if it does not exist.
    Prevents errors when saving cleaned datasets.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[AUTO] Created folder: {path}")


# ----------------------------------------------------------------------
# Helper: Clean text function
# ----------------------------------------------------------------------
def clean_text(text):
    """
    Cleans a single blog post using several NLP preprocessing steps.

    Steps include:
    - Lowercasing text
    - Removing punctuation
    - Removing numbers
    - Removing stopwords
    - Tokenizing text

    Returns:
        Cleaned text string
    """

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation using regex
    text = re.sub(r"[^\w\s]", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Tokenize the text into words
    tokens = word_tokenize(text)

    # Load English stopwords
    stop_words = set(stopwords.words("english"))

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Join tokens back into a string
    cleaned_text = " ".join(tokens)

    return cleaned_text


# ----------------------------------------------------------------------
# Main Preprocessing Function
# ----------------------------------------------------------------------
def preprocess_dataset(df):
    """
    Cleans the entire DataFrame by applying clean_text() to each row.

    Parameters:
        df (DataFrame): raw dataset loaded from load_data.py.

    Returns:
        cleaned_df (DataFrame)
    """

    print("[INFO] Starting preprocessing...")

    # Drop rows where Data column is empty
    df = df.dropna(subset=["Data"])

    # Apply cleaning to each blog post
    print("[INFO] Cleaning text... This may take a few seconds.")
    df["Cleaned_Text"] = df["Data"].apply(clean_text)

    print("[INFO] Text cleaning completed.")
    print(f"[INFO] New dataset shape: {df.shape}")

    return df


# ----------------------------------------------------------------------
# Function: Save the cleaned dataset
# ----------------------------------------------------------------------
def save_cleaned_dataset(df, path="data/processed/cleaned_blogs.csv"):
    """
    Saves the cleaned DataFrame to the processed folder.
    """
    ensure_processed_folder("data/processed/")
    df.to_csv(path, index=False)
    print(f"[INFO] Cleaned dataset saved at: {path}")


# ----------------------------------------------------------------------
# Self-Test Block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from load_data import load_raw_dataset

    print("[TEST] Running preprocess.py directly...")

    # Load raw dataset
    df_raw = load_raw_dataset()

    if df_raw is not None:
        # Preprocess it
        df_clean = preprocess_dataset(df_raw)

        # Save cleaned file
        save_cleaned_dataset(df_clean)

        print("[TEST] Preprocessing test completed successfully.")
