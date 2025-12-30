import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import os

# -------------------------------------------------------------
# Function: handle_missing_values
# Purpose : Clean dataset by handling missing values properly.
# -------------------------------------------------------------
def handle_missing_values(df):
    """
    Handles missing values in the dataset.

    Numerical columns  -> filled with median (robust to outliers)
    Categorical columns -> filled with mode (most frequent value)

    This method prevents dropping rows unnecessarily.
    """
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:  
            # For numerical data, use median (better than mean for skewed data)
            df[column].fillna(df[column].median(), inplace=True)
        else:
            # For categorical data, use mode (most frequent category)
            df[column].fillna(df[column].mode()[0], inplace=True)

    return df


# -------------------------------------------------------------
# Function: encode_features
# Purpose : Apply One-Hot Encoding or Label Encoding based on rule:
#           - Categorical columns < 5 unique values → One-Hot Encode
#           - Categorical columns ≥ 5 unique values → Label Encode
# -------------------------------------------------------------
def encode_features(df):
    """
    Encodes categorical columns using the appropriate encoding technique.

    Returns:
        df (DataFrame) - encoded DataFrame
    """
    categorical_cols = [col for col in df.select_dtypes(include=['object']).columns if col != 'income']

    for col in categorical_cols:
        unique_count = df[col].nunique()

        # If the category count is small, do One-Hot Encoding
        if unique_count < 5:
            df = pd.get_dummies(df, columns=[col], prefix=col)

        else:
            # If many categories, use Label Encoding
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])

    return df


# -------------------------------------------------------------
# Function: scale_features
# Purpose : Apply StandardScaler and MinMaxScaler to numerical columns.
#           This normalizes the data, improving ML model performance.
# -------------------------------------------------------------
def scale_features(df):
    """
    Scales numerical columns using StandardScaler and MinMaxScaler.

    Returns:
        df (DataFrame) - DataFrame with scaled features
    """

    numeric_cols = [
    col for col in df.select_dtypes(include=['int64', 'float64']).columns
    if col != 'income'
]


    # Create scaler objects
    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    # Create new scaled columns
    df[numeric_cols] = standard_scaler.fit_transform(df[numeric_cols])
    df[numeric_cols] = minmax_scaler.fit_transform(df[numeric_cols])

    return df


# -------------------------------------------------------------
# Function: preprocess_data
# Purpose : Calls all preprocessing steps:
#           missing values → encoding → scaling → save cleaned CSV
# -------------------------------------------------------------
def preprocess_data(df, save_path="data/processed/cleaned_data.csv"):
    """
    Runs the full preprocessing pipeline on the dataset.
    """

    print("[INFO] Handling missing values...")
    df = handle_missing_values(df)

    print("[INFO] Encoding categorical features...")
    df = encode_features(df)

    print("[INFO] Scaling numerical features...")
    df = scale_features(df)

    # Ensure output directory exists
    os.makedirs("data/processed", exist_ok=True)

    # Save cleaned preprocessed dataset
    df.to_csv(save_path, index=False)
    print(f"[INFO] Cleaned dataset saved to: {save_path}")

    return df


