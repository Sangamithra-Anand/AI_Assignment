"""
preprocessing.py
----------------
This file is responsible for:
1. Loading the Titanic dataset
2. Cleaning the dataset
3. Handling missing values
4. Encoding categorical variables
5. Creating new useful features
6. Saving the cleaned dataset for model training

FIRST major step in the ML pipeline.
"""

import pandas as pd
import os


def load_data(path):
    """Loads CSV dataset from given path."""
    return pd.read_csv(path)


def clean_data(df):
    """
    Perform ALL cleaning steps:
    - Handle missing values
    - Drop useless columns
    - Encode categories
    - Create new features
    - Ensure no NaN values remain
    """

    # ----------------------------------------
    # 1. Handle Missing Values
    # ----------------------------------------

    # Age → fill median
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Embarked → fill most common
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Drop heavily-missing columns
    df = df.drop(columns=["Cabin", "Ticket", "Name"])

    # ----------------------------------------
    # 2. Create new feature: FamilySize
    # ----------------------------------------
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # ----------------------------------------
    # 3. Encode Categorical Columns
    # ----------------------------------------
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    # ----------------------------------------
    # 4. Drop unneeded column
    # ----------------------------------------
    df = df.drop(columns=["PassengerId"])

    # ----------------------------------------
    # 5. FINAL SAFETY STEP (VERY IMPORTANT)
    #    Remove ANY remaining NaN values
    # ----------------------------------------
    df = df.fillna(0)      # <-- Fixes NaN error during model training

    return df


def save_clean_data(df, output_path):
    """Save cleaned DataFrame to CSV."""

    # AUTO-CREATE output directory if missing
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_path, index=False)


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "Titanic_train.csv")
    OUTPUT_PATH = os.path.join(BASE_DIR, "output", "clean_train.csv")

    print("[INFO] Loading dataset...")
    df = load_data(DATA_PATH)

    print("[INFO] Cleaning dataset...")
    clean_df = clean_data(df)

    print("[INFO] Saving cleaned dataset...")
    save_clean_data(clean_df, OUTPUT_PATH)

    print("[DONE] Preprocessing complete!")
    print(f"[SAVED] {OUTPUT_PATH}")
