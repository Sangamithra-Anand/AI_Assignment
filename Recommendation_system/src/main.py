"""
main.py
--------
Main menu controller for the Anime Recommendation System.
UPDATED:
- Removed similarity matrix generation
- Uses on-demand similarity in recommend.py
"""

import sys
import pickle
import pandas as pd

from utils.helpers import log_message, start_timer, end_timer

from src.load_data import load_raw_dataset
from src.preprocess import preprocess_dataset, save_cleaned_dataset
from src.feature_engineering import extract_features, save_feature_artifacts
from src.recommend import recommend_anime, load_cleaned_data
from src.evaluation import evaluate_system


# -------------------------------------------------------------
# Run FULL PIPELINE
# -------------------------------------------------------------
def run_full_pipeline():
    log_message("Running FULL PIPELINE...", "INFO")
    t0 = start_timer()

    # 1. Load raw
    df_raw = load_raw_dataset()
    if df_raw is None:
        return

    # 2. Preprocess
    df_clean = preprocess_dataset(df_raw)
    save_cleaned_dataset(df_clean)

    # 3. Feature extraction
    config, matrix = extract_features(df_clean)
    save_feature_artifacts(config, matrix)

    end_timer(t0, "FULL PIPELINE")
    log_message("Pipeline completed ✔", "INFO")


# -------------------------------------------------------------
# Menu
# -------------------------------------------------------------
def menu():
    while True:
        print("\n")
        print("=" * 60)
        print("          ANIME RECOMMENDATION SYSTEM — MENU")
        print("=" * 60)
        print("1. Load Raw Dataset")
        print("2. Preprocess Dataset")
        print("3. Feature Engineering (Custom TF-IDF + Numeric)")
        print("4. Compute Cosine Similarity Matrix")
        print("5. ⭐⭐  GET RECOMMENDATIONS (HIGHLY RECOMMENDED) ⭐⭐")
        print("6. Evaluate System (Precision | Recall | F1)")
        print("7. Run FULL PIPELINE")
        print("8. Exit")
        print("=" * 60)
        choice = input("Enter your choice (1–8): ").strip()

        if choice == "1":
            load_raw_dataset()

        elif choice == "2":
            df_raw = load_raw_dataset()
            if df_raw is not None:
                df_clean = preprocess_dataset(df_raw)
                save_cleaned_dataset(df_clean)

        elif choice == "3":
            df_clean = load_cleaned_data()
            if df_clean is not None:
                config, matrix = extract_features(df_clean)
                save_feature_artifacts(config, matrix)

        elif choice == "5":
            anime = input("Enter anime title: ")
            result = recommend_anime(anime)
            if result is not None:
                print(result)

        elif choice == "6":
            df_clean = load_cleaned_data()
            if df_clean is not None:
                print(evaluate_system(df_clean))

        elif choice == "7":
            run_full_pipeline()

        elif choice == "8":
            log_message("Goodbye!", "INFO")
            sys.exit()

        else:
            log_message("Invalid choice.", "WARNING")


if __name__ == "__main__":
    log_message("Starting Anime Recommendation System...", "INFO")
    menu()
