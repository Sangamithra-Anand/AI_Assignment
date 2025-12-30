"""
recommend.py
-------------
This file computes anime recommendations.

MAJOR IMPROVEMENT:
------------------
We NO LONGER load a giant similarity matrix.
Instead, we compute similarity ONLY for the selected anime:

    similarity = feature_matrix @ feature_matrix[idx]

This is:
- SUPER FAST
- SUPER MEMORY EFFICIENT
"""

import os
import pickle
import numpy as np
import pandas as pd

# utils folder is OUTSIDE src, so use direct import
from utils.helpers import log_message, start_timer, end_timer


# -------------------------------------------------------------
# Load cleaned dataset
# -------------------------------------------------------------
def load_cleaned_data(path="data/processed/cleaned_anime.csv"):
    if not os.path.exists(path):
        log_message(f"Missing cleaned dataset: {path}", "ERROR")
        return None

    df = pd.read_csv(path)
    log_message("Loaded cleaned dataset.", "INFO")
    return df


# -------------------------------------------------------------
# Load normalized feature matrix
# -------------------------------------------------------------
def load_feature_matrix(path="data/processed/features_matrix.pkl"):
    if not os.path.exists(path):
        log_message("Feature matrix missing. Run Feature Engineering.", "ERROR")
        return None

    with open(path, "rb") as f:
        matrix = pickle.load(f)

    return matrix


# -------------------------------------------------------------
# Recommend anime using ON-THE-FLY cosine similarity
# -------------------------------------------------------------
def recommend_anime(name, top_n=10):

    log_message(f"Computing recommendations for '{name}'...", "INFO")
    t0 = start_timer()

    # 1. Load data
    df = load_cleaned_data()
    matrix = load_feature_matrix()

    if df is None or matrix is None:
        return None

    # 2. Validate title
    if name not in df["name"].values:
        log_message(f"'{name}' not found.", "ERROR")
        matches = df[df["name"].str.contains(name[:3], case=False)]
        if len(matches): print(matches["name"].head())
        return None

    idx = df.index[df["name"] == name][0]

    # ---------------------------------------------------------
    # 3. Compute cosine similarity on-the-fly
    # ---------------------------------------------------------
    """
    Since the feature matrix is already normalized:
        dot(a, b) == cosine_similarity(a, b)
    """
    base = matrix[idx]                     # shape (d,)
    similarity = matrix @ base             # shape (N,) → fast

    # Sort descending
    scores = list(enumerate(similarity))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Skip itself
    scores = scores[1:top_n+1]

    ids = [i[0] for i in scores]
    sims = [i[1] for i in scores]

    result = pd.DataFrame({
        "Recommended Anime": df.loc[ids, "name"],
        "Genre": df.loc[ids, "genre"],
        "Rating": df.loc[ids, "rating"],
        "Similarity Score": sims
    })

    end_timer(t0, "Recommendation Completed")

    return result
