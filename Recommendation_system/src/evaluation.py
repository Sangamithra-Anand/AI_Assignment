"""
evaluation.py
--------------
This file evaluates the recommendation system using:

- Precision@K
- Recall@K
- F1-Score@K

UPDATED VERSION:
- Uses logging, timers, and standardized output
- Cleaner evaluation loop
"""

import pandas as pd
from src.recommend import recommend_anime
from utils.helpers import log_message, start_timer, end_timer


# -------------------------------------------------------------
# Convert genre string → Set
# -------------------------------------------------------------
def get_genre_set(genre_string):
    """
    Converts "Action, Comedy, Fantasy"
    → {"action", "comedy", "fantasy"}

    Used for comparing genre overlap.
    """
    return set(g.strip().lower() for g in genre_string.split(",") if g.strip() != "")



# -------------------------------------------------------------
# Evaluate Recommendation System
# -------------------------------------------------------------
def evaluate_system(df, top_k=10, sample_size=50):
    """
    Evaluates recommendation quality using:
        Precision@K
        Recall@K
        F1 Score@K

    Parameters:
        df (DataFrame): Cleaned anime dataset
        top_k (int): Number of recommendations per anime
        sample_size (int): Number of anime to evaluate

    Evaluation Method:
    - Use GENRE OVERLAP as a proxy for ground truth similarity.
    """

    log_message("Starting system evaluation...", "INFO")
    t0 = start_timer()

    # Random sampling to speed up evaluation
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)

    total_correct = 0
    total_relevant = 0
    total_recommended = top_k * len(sample_df)

    count = 0

    for _, row in sample_df.iterrows():
        anime_name = row["name"]
        true_genres = get_genre_set(row["genre"])

        log_message(f"Evaluating anime: {anime_name}", "INFO")

        recommendations = recommend_anime(anime_name, top_n=top_k)

        if recommendations is None:
            log_message(f"Skipping {anime_name} — no recommendations returned.", "WARNING")
            continue

        correct_count = 0
        relevant_items = 0

        # Compare genre overlap
        for _, rec in recommendations.iterrows():
            rec_genres = get_genre_set(rec["Genre"])
            if len(rec_genres.intersection(true_genres)) > 0:
                relevant_items += 1
                correct_count += 1

        total_correct += correct_count
        total_relevant += max(relevant_items, 1)  # avoid divide-by-zero

        count += 1

    # ---------------------------------------------------------
    # METRIC CALCULATION
    # ---------------------------------------------------------
    precision = total_correct / total_recommended if total_recommended else 0
    recall = total_correct / total_relevant if total_relevant else 0

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    results = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4)
    }

    end_timer(t0, "Evaluation")

    log_message(f"Evaluation Completed. Precision: {results['precision']} | "
                f"Recall: {results['recall']} | F1 Score: {results['f1_score']}", "INFO")

    return results



# -------------------------------------------------------------
# TEST BLOCK
# -------------------------------------------------------------
if __name__ == "__main__":
    log_message("Running evaluation.py test mode...", "INFO")

    from src.recommend import load_cleaned_data
    df = load_cleaned_data()

    if df is None:
        log_message("ERROR: Cleaned dataset is missing. Cannot run evaluation.", "ERROR")
        exit()

    metrics = evaluate_system(df, top_k=5, sample_size=20)

    print("\nFinal Evaluation Metrics:")
    print(metrics)

    log_message("Evaluation Test Completed.", "INFO")


