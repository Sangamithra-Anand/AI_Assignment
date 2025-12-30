# ------------------------------------------------------------
# PREPROCESSING SCRIPT
# ------------------------------------------------------------
# This script:
# 1. Cleans missing values
# 2. Encodes categorical columns
# 3. Splits training data into train/validation sets
# 4. Returns processed data for model training
#
# It MUST accept: preprocess_data(train_df, test_df)
# ------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ------------------------------------------------------------
# CLEAN MISSING VALUES
# ------------------------------------------------------------
def handle_missing_values(df):
    """
    Handles missing values in both train & test datasets.
    """

    # Fill missing Age with median
    df["Age"].fillna(df["Age"].median(), inplace=True)

    # Fill missing Fare with median
    df["Fare"].fillna(df["Fare"].median(), inplace=True)

    # Fill missing Embarked with mode
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # Cabin has MANY missing → drop it
    if "Cabin" in df.columns:
        df.drop("Cabin", axis=1, inplace=True)

    return df


# ------------------------------------------------------------
# LABEL ENCODING (Sex, Embarked)
# ------------------------------------------------------------
def encode_features(df):
    """
    Converts categorical columns into numbers.
    """
    label = LabelEncoder()

    df["Sex"] = label.fit_transform(df["Sex"])          # male=1, female=0
    df["Embarked"] = label.fit_transform(df["Embarked"])

    return df


# ------------------------------------------------------------
# MAIN PREPROCESS FUNCTION
# ------------------------------------------------------------
def preprocess_data(train_df, test_df):
    """
    FULL preprocessing pipeline.
    Accepts:
        train_df (DataFrame)
        test_df (DataFrame)

    Returns:
        X_train, y_train, X_val, y_val, X_test
    """

    # Step 1 — handle missing values
    train_df = handle_missing_values(train_df)
    test_df = handle_missing_values(test_df)

    # Step 2 — encode categorical columns
    train_df = encode_features(train_df)
    test_df = encode_features(test_df)

    # Step 3 — Select features for model training
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

    X = train_df[features]
    y = train_df["Survived"]

    X_test = test_df[features]

    # Step 4 — Split train → train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, y_train, X_val, y_val, X_test


