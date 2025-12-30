"""
MAIN PIPELINE SCRIPT
---------------------

This file controls the entire machine learning preprocessing workflow.

Steps performed:
1. Load the dataset
2. Run preprocessing (missing values, encoding, scaling)
3. Run feature engineering (new features + log transformation)
4. Run feature selection (Isolation Forest + PPS matrix)
5. Run visualizations (Correlation + PPS heatmaps)
6. Save reports

Every step prints clear messages so you know what is happening.
"""

# ----------------------- IMPORT MODULES -----------------------

from src.load_data import load_dataset
from src.preprocess import preprocess_data
from src.feature_engineering import engineer_features
from src.feature_selection import feature_selection_pipeline
from src.visualization import visualization_pipeline
from src.utils import save_text_report, print_section


# =============================================================
#                    MAIN EXECUTION FUNCTION
# =============================================================
def main():
    """Runs the entire data processing pipeline step by step."""

    # ---------------------------------------------------------
    # STEP 1: LOAD DATASET
    # ---------------------------------------------------------
    print_section("STEP 1: LOADING DATASET")
    df = load_dataset()   # Loads from data/raw/adult_with_headers.csv

    # Save basic dataset summary as a report
    summary_text = (
        "DATASET SUMMARY REPORT\n"
        "-----------------------\n"
        f"Shape: {df.shape}\n\n"
        f"Columns:\n{df.dtypes}\n\n"
        f"Missing Values:\n{df.isnull().sum()}\n"
    )
    save_text_report(summary_text, "data/reports/eda_report.txt")



    # ---------------------------------------------------------
    # STEP 2: PREPROCESSING
    # Missing values → Encoding → Scaling
    # ---------------------------------------------------------
    print_section("STEP 2: PREPROCESSING DATA")
    df_preprocessed = preprocess_data(df)



    # ---------------------------------------------------------
    # STEP 3: FEATURE ENGINEERING
    # Create new features + log transformation
    # ---------------------------------------------------------
    print_section("STEP 3: FEATURE ENGINEERING")
    df_engineered = engineer_features(df_preprocessed)



    # ---------------------------------------------------------
    # STEP 4: FEATURE SELECTION
    # Isolation Forest → Remove outliers → PPS matrix
    # ---------------------------------------------------------
    print_section("STEP 4: FEATURE SELECTION")
    df_selected, pps_matrix = feature_selection_pipeline(df_engineered)



    # ---------------------------------------------------------
    # STEP 5: VISUALIZATION (Correlation + PPS Heatmaps)
    # ---------------------------------------------------------
    print_section("STEP 5: CREATING VISUALIZATIONS")
    visualization_pipeline(df_selected)



    # ---------------------------------------------------------
    # PIPELINE COMPLETE
    # ---------------------------------------------------------
    print_section("PIPELINE COMPLETED SUCCESSFULLY")
    print("[INFO] All tasks finished. Check 'output/' and 'data/processed/' folders for results.")



# =============================================================
#                  RUN THE MAIN FUNCTION
# =============================================================
if __name__ == "__main__":
    main()


