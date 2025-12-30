# ---------------------------------------------------------------
# EDA MODULE - Exploratory Data Analysis
# ---------------------------------------------------------------
# This module handles all the basic dataset exploration tasks:
#
#   1. Loading the dataset
#   2. Showing first 5 rows
#   3. Showing dataset shape
#   4. Displaying dataset structure and data types
#   5. Checking missing values
#   6. Displaying class distribution (edible vs poisonous)
#
# These functions are imported and used inside main.py
# ---------------------------------------------------------------

import pandas as pd


# ---------------------------------------------------------------
# FUNCTION: load_data()
# ---------------------------------------------------------------
# PURPOSE:
#   Load the mushroom dataset from a given file path.
#
# RETURNS:
#   A pandas DataFrame containing the dataset.
# ---------------------------------------------------------------
def load_data(path):
    print(f"\nLoading dataset from: {path}")
    df = pd.read_csv(path)
    return df


# ---------------------------------------------------------------
# FUNCTION: show_basic_info()
# ---------------------------------------------------------------
# PURPOSE:
#   Display:
#       - first 5 rows
#       - shape (rows, columns)
#       - overall dataset info (column types, null counts)
# ---------------------------------------------------------------
def show_basic_info(df):
    print("\n--- FIRST 5 ROWS ---")
    print(df.head())

    print("\n--- SHAPE OF DATASET ---")
    print(df.shape)

    print("\n--- DATASET INFO ---")
    print(df.info())


# ---------------------------------------------------------------
# FUNCTION: check_missing_values()
# ---------------------------------------------------------------
# PURPOSE:
#   Check for missing (null) values in each column.
#
# RETURNS:
#   Prints the number of missing values per column.
# ---------------------------------------------------------------
def check_missing_values(df):
    print("\n--- MISSING VALUES IN EACH COLUMN ---")
    print(df.isnull().sum())


# ---------------------------------------------------------------
# FUNCTION: class_distribution()
# ---------------------------------------------------------------
# PURPOSE:
#   Show how many mushrooms are:
#       - edible (e)
#       - poisonous (p)
#
# WHY:
#   This helps us check if the dataset is balanced.
# ---------------------------------------------------------------
def class_distribution(df):
    print("\n--- CLASS DISTRIBUTION (edible vs poisonous) ---")
    print(df['class'].value_counts())


