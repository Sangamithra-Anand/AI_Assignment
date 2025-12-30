"""
preprocess.py
--------------
This file handles all preprocessing steps for the Glass Dataset.

Tasks performed:
1. Create 'data/processed' folder automatically if missing
2. Remove duplicates (if any)
3. Handle missing values (mean/median imputation)
4. Apply feature scaling (StandardScaler)
5. Handle class imbalance using SMOTE
6. Save the cleaned dataset into: data/processed/cleaned_glass.csv

Notes:
- Random Forest does NOT require scaling, but we scale anyway for
  consistency and because Boosting models benefit from scaled data.
- SMOTE is applied only if there is imbalance.
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# ---------------------------------------------------------------------
# Helper Function: Ensure processed folder exists (AUTO-CREATION)
# ---------------------------------------------------------------------
def ensure_processed_folder(path="data/processed/"):
    """
    Creates 'data/processed' folder automatically if it doesn't exist.

    Explanation:
    ------------
    - Many beginners forget to manually create folders.
    - To avoid errors when saving cleaned files, we create the folder
      inside the code itself.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[AUTO] Created folder: {path}")


# ---------------------------------------------------------------------
# Main Function: Preprocess Dataset
# ---------------------------------------------------------------------
def preprocess_glass_data(df):
    """
    This function performs all preprocessing steps.

    Parameters:
    -----------
    df : pandas.DataFrame
         Raw dataset loaded from load_data.py

    Returns:
    --------
    final_df : pandas.DataFrame
               Cleaned and preprocessed dataset

    Explanation of Steps:
    ---------------------
    1. Remove duplicates (good practice)
    2. Handle missing values
    3. Separate features (X) and target (y)
    4. Apply scaling
    5. Handle imbalance using SMOTE
    6. Recombine scaled + resampled data
    7. Save to 'data/processed/'
    """

    print("\n[INFO] Starting preprocessing...")

    # ---------------------------------------------------------
    # 1. Remove duplicate rows
    # ---------------------------------------------------------
    duplicates = df.duplicated().sum()
    print(f"[INFO] Duplicate rows found: {duplicates}")

    df = df.drop_duplicates()
    print("[INFO] Duplicate rows removed.")

    # ---------------------------------------------------------
    # 2. Handle missing values
    # ---------------------------------------------------------
    print("[INFO] Checking for missing values...")
    missing = df.isnull().sum()
    print(missing)

    if missing.sum() > 0:
        print("[INFO] Missing values detected → Applying mean imputation.")
        df = df.fillna(df.mean())
    else:
        print("[INFO] No missing values found.")

    # ---------------------------------------------------------
    # 3. Separate features & target
    # ---------------------------------------------------------
    print("[INFO] Splitting features (X) and target (y).")

    # Assumption: Last column is the target variable (Glass Type)
    X = df.iloc[:, :-1]     # All columns except last
    y = df.iloc[:, -1]      # Last column

    # ---------------------------------------------------------
    # 4. Feature Scaling (Standardization)
    # ---------------------------------------------------------
    print("[INFO] Applying StandardScaler to features...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert scaled numpy array back to DataFrame
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # ---------------------------------------------------------
    # 5. Handle Imbalance using SMOTE
    # ---------------------------------------------------------
    print("[INFO] Checking for class imbalance...")

    class_counts = y.value_counts()
    print(class_counts)

    # If the smallest class has fewer than 20% of the largest → imbalance
    imbalance_ratio = class_counts.min() / class_counts.max()

    if imbalance_ratio < 0.5:  # threshold
        print("[INFO] Imbalance detected → Applying SMOTE oversampling.")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
        print("[INFO] SMOTE applied successfully!")

    else:
        print("[INFO] No severe imbalance detected → Skipping SMOTE.")
        X_resampled, y_resampled = X_scaled, y

    # ---------------------------------------------------------
    # 6. Recombine features + target
    # ---------------------------------------------------------
    final_df = pd.concat([X_resampled, y_resampled], axis=1)
    print("[INFO] Dataset recombined into final form.")

    # ---------------------------------------------------------
    # 7. Save cleaned dataset
    # ---------------------------------------------------------
    ensure_processed_folder()

    save_path = "data/processed/cleaned_glass.csv"
    final_df.to_csv(save_path, index=False)
    print(f"[INFO] Cleaned dataset saved to: {save_path}")

    return final_df


# ---------------------------------------------------------------------
# TEST BLOCK: Run this file alone → python src/preprocess.py
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("[TEST] Running preprocess.py directly...")

    try:
        temp_df = pd.read_excel("data/raw/glass.xlsx")
        processed = preprocess_glass_data(temp_df)
        print("\n[TEST] preprocess.py is working correctly ✔️")
    except Exception as e:
        print(f"[TEST ERROR] {e}")
