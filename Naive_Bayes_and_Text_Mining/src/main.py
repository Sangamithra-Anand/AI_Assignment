"""
main.py
----------------------------------------------
This is the MAIN PIPELINE CONTROLLER for the:

    BLOG TEXT CLASSIFICATION + SENTIMENT ANALYSIS PROJECT

Features:
1. Load Raw Data
2. Preprocess Data
3. Train Naive Bayes Model (TF-IDF included)
4. Evaluate Model
5. Perform Sentiment Analysis
6. Run ALL STEPS in correct order

This file gives a clean console menu to run the entire system step-by-step.
"""

import sys
from load_data import load_raw_dataset
from preprocess import preprocess_dataset, save_cleaned_dataset
from feature_engineering import load_cleaned_dataset
from train_model import train_naive_bayes
from evaluate import evaluate_model
from sentiment_analysis import analyze_sentiment, save_sentiment_results


# -----------------------------------------------------------
# Helper Function: pause after each task
# -----------------------------------------------------------
def pause():
    input("\nPress ENTER to return to menu...")


# -----------------------------------------------------------
# Option 1: Load Raw Data
# -----------------------------------------------------------
def option_load_data():
    print("\n=====================================")
    print("        LOADING RAW DATASET")
    print("=====================================")

    df = load_raw_dataset()
    if df is not None:
        print("\n[INFO] RAW DATA PREVIEW:")
        print(df.head())

    pause()


# -----------------------------------------------------------
# Option 2: Preprocess Data
# -----------------------------------------------------------
def option_preprocess_data():
    print("\n=====================================")
    print("        PREPROCESSING DATA")
    print("=====================================")

    df = load_raw_dataset()
    if df is None:
        print("[ERROR] Dataset missing! Add blogs_categories.csv in data/raw/")
        pause()
        return

    cleaned_df = preprocess_dataset(df)
    save_cleaned_dataset(cleaned_df)

    pause()


# -----------------------------------------------------------
# Option 3: Train Naive Bayes Model
# -----------------------------------------------------------
def option_train_model():
    print("\n=====================================")
    print("      TRAINING NAIVE BAYES MODEL")
    print("=====================================")

    train_naive_bayes()

    pause()


# -----------------------------------------------------------
# Option 4: Evaluate Model
# -----------------------------------------------------------
def option_evaluate_model():
    print("\n=====================================")
    print("        EVALUATING MODEL")
    print("=====================================")

    evaluate_model()

    pause()


# -----------------------------------------------------------
# Option 5: Perform Sentiment Analysis
# -----------------------------------------------------------
def option_sentiment_analysis():
    print("\n=====================================")
    print("        SENTIMENT ANALYSIS")
    print("=====================================")

    df = load_cleaned_dataset()
    if df is None:
        print("[ERROR] You must run preprocessing before sentiment analysis!")
        pause()
        return

    sentiment_df = analyze_sentiment(df)
    save_sentiment_results(sentiment_df)

    pause()


# -----------------------------------------------------------
# Option 6: Run FULL PIPELINE Automatically
# -----------------------------------------------------------
def option_run_all():
    print("\n=====================================")
    print("         RUNNING FULL PIPELINE")
    print("=====================================")

    # Step 1: Load Data
    df = load_raw_dataset()
    if df is None:
        print("[ERROR] Dataset missing! Add blogs_categories.csv in data/raw/")
        pause()
        return

    # Step 2: Preprocess Data
    cleaned_df = preprocess_dataset(df)
    save_cleaned_dataset(cleaned_df)

    # Step 3: Train Model
    train_naive_bayes()

    # Step 4: Evaluate Model
    evaluate_model()

    # Step 5: Sentiment Analysis
    sentiment_df = analyze_sentiment(cleaned_df)
    save_sentiment_results(sentiment_df)

    print("\n[INFO] FULL PIPELINE EXECUTED SUCCESSFULLY!")

    pause()


# -----------------------------------------------------------
# MENU FUNCTION
# -----------------------------------------------------------
def menu():
    while True:
        print("\n=====================================")
        print("      BLOG CLASSIFICATION SYSTEM")
        print("=====================================")
        print("1. Load Raw Dataset")
        print("2. Preprocess Dataset")
        print("3. Train Naive Bayes Model")
        print("4. Evaluate Model")
        print("5. Perform Sentiment Analysis")
        print("6. RUN FULL PIPELINE (Recommended)")
        print("7. Exit")
        print("=====================================")

        choice = input("Enter your choice (1-7): ").strip()

        if choice == "1":
            option_load_data()

        elif choice == "2":
            option_preprocess_data()

        elif choice == "3":
            option_train_model()

        elif choice == "4":
            option_evaluate_model()

        elif choice == "5":
            option_sentiment_analysis()

        elif choice == "6":
            option_run_all()

        elif choice == "7":
            print("Exiting program... Goodbye!")
            sys.exit()

        else:
            print("[ERROR] Invalid option. Please choose between 1 and 7.")


# -----------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------
if __name__ == "__main__":
    menu()
