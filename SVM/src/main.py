# ---------------------------------------------------------------
# MAIN EXECUTION FILE
# ---------------------------------------------------------------
# This file controls the entire machine learning pipeline:
#
#   1. Load the dataset
#   2. Perform Exploratory Data Analysis (EDA)
#   3. Visualize important features
#   4. Preprocess the dataset (Label Encoding + Split)
#   5. Train the SVM model
#   6. Evaluate the model performance
#   7. Save the trained model
#
# All functions used here come from:
#   - eda.py
#   - visualize.py
#   - preprocess.py
#   - train_svm.py
#
# Run using:
#   python src/main.py
# ---------------------------------------------------------------

import os
import pandas as pd

# Import our custom modules
from eda import load_data, show_basic_info, check_missing_values, class_distribution
from visualize import (
    plot_class_distribution,
    plot_categorical_vs_class,
    plot_correlation_matrix,
    plot_top_correlated_features,
)
from preprocess import encode_categorical, split_data
from train_svm import train_svm_model, evaluate_model, save_svm_model


# ---------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------
def main():
    print("\n==============================================")
    print("      MUSHROOM CLASSIFICATION USING SVM")
    print("==============================================\n")

    # ----------------------------
    # 1. LOAD THE DATASET
    # ----------------------------
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "mushroom.csv")
    data_path = os.path.normpath(data_path)
    df = load_data(data_path)

    print("\n--- DATA LOADED SUCCESSFULLY ---")

    # ----------------------------
    # 2. BASIC EDA
    # ----------------------------
    show_basic_info(df)
    check_missing_values(df)
    class_distribution(df)

    # ----------------------------
    # 3. VISUALIZATION (RAW DATA)
    # ----------------------------
    print("\n--- CREATING VISUALIZATIONS ---")
    plot_class_distribution(df)
    plot_categorical_vs_class(df, 'odor', rotate_x=True)
    plot_categorical_vs_class(df, 'habitat', rotate_x=True)

    # ----------------------------
    # 4. PREPROCESSING
    # ----------------------------
    print("\n--- ENCODING CATEGORICAL DATA ---")
    df_encoded = encode_categorical(df.copy())  # Use a copy to preserve original df

    # Plot correlation graph only on numeric data
    plot_correlation_matrix(df_encoded)
    plot_top_correlated_features(df_encoded, target_col='class')

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df_encoded)

    # ----------------------------
    # 5. TRAINING THE SVM MODEL
    # ----------------------------
    svm_model = train_svm_model(X_train, y_train, kernel='rbf')

    # ----------------------------
    # 6. EVALUATE MODEL PERFORMANCE
    # ----------------------------
    evaluate_model(svm_model, X_test, y_test)

    # ----------------------------
    # 7. SAVE THE TRAINED MODEL
    # ----------------------------
    save_svm_model(svm_model)

    print("\n==============================================")
    print("         PIPELINE EXECUTION COMPLETED")
    print("==============================================\n")


# ---------------------------------------------------------------
# RUN THE MAIN FUNCTION
# ---------------------------------------------------------------
if __name__ == "__main__":
    main()
