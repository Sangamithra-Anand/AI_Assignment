# ---------------------------------------------------------------
# PREPROCESSING MODULE
# ---------------------------------------------------------------
# This file is responsible for preparing the mushroom dataset
# before training the SVM model.
#
# Steps handled by this module:
#   1. Convert categorical (text) values → numerical (Label Encoding)
#   2. Split the dataset into Training and Testing sets
#
# NOTE:
#   Machine Learning algorithms (like SVM) CANNOT understand text.
#   They only work with numerical values.
#   So encoding is a mandatory preprocessing step.
# ---------------------------------------------------------------

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------
# FUNCTION: encode_categorical(df)
# ---------------------------------------------------------------
# PURPOSE:
#   Convert ALL columns from text → numbers using Label Encoding.
#
# HOW IT WORKS:
#   - The mushroom dataset is fully categorical (every column is text).
#   - LabelEncoder replaces each unique text value with a number.
#   - Example: odor column → {'foul': 0, 'almond': 1, 'none': 2, ...}
#
# IMPORTANT:
#   This modifies the dataframe directly and returns the updated one.
# ---------------------------------------------------------------
def encode_categorical(df):

    # Create a new LabelEncoder object
    # We will reuse this encoder for every column.
    label_encoder = LabelEncoder()

    # Loop through every column in the dataset
    for col in df.columns:

        # Replace the column values with encoded (numeric) values.
        # fit_transform() identifies all unique text values and maps them.
        df[col] = label_encoder.fit_transform(df[col])

        # Show the mapping created for this column (for understanding)
        # Example output: {'e':0, 'p':1}
        print(f"Encoding for column '{col}': {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    # Return the encoded dataframe
    return df



# ---------------------------------------------------------------
# FUNCTION: split_data(df)
# ---------------------------------------------------------------
# PURPOSE:
#   Split the encoded dataset into:
#       - X_train  (training features)
#       - X_test   (testing features)
#       - y_train  (training labels)
#       - y_test   (testing labels)
#
# WHAT ARE X AND Y?
#   X = all features (input)
#   Y = the target/label ('class' column → edible or poisonous)
#
# WHY SPLIT?
#   - Model learns on training data (X_train, y_train)
#   - Model is tested on new unseen data (X_test, y_test)
#   - This prevents cheating and overfitting
#
# test_size = 0.2 → 20% of data used for testing
# random_state = 42 → ensures the same random split every run
# ---------------------------------------------------------------
def split_data(df):

    # Separate features (X) and target (y)
    X = df.drop("class", axis=1)   # drop the 'class' column → this becomes X (input)
    y = df["class"]                # only the 'class' column → this becomes y (output label)

    # Split the dataset:
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2,            # 20% test data
        random_state=42           # makes results repeatable
    )

    # Print shapes so you understand the split clearly
    print("\n--- TRAIN-TEST SPLIT DETAILS ---")
    print("Training Features Shape (X_train):", X_train.shape)
    print("Testing Features Shape  (X_test):", X_test.shape)
    print("Training Labels Shape   (y_train):", y_train.shape)
    print("Testing Labels Shape    (y_test):", y_test.shape)

    return X_train, X_test, y_train, y_test
