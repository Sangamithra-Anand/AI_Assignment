"""
feature_engineering.py
----------------------
This file builds the FEATURE MATRIX used for recommendation.

CHANGES (Python 3.13 compatible):
---------------------------------
1. Custom TF-IDF-like encoding for genres (no sklearn needed)
2. Manual numeric scaling using NumPy
3. NORMALIZES the final feature matrix so cosine similarity can be
   computed using a simple dot product → FAST and memory efficient.
"""

import os
import pickle
import numpy as np
import pandas as pd

# We import helper functions from utils folder (sibling of src)
from utils.helpers import ensure_folder, log_message, start_timer, end_timer


# -------------------------------------------------------------
# Convert genre strings into sets of tags
# -------------------------------------------------------------
def _genre_to_set(genre_str: str):
    """
    Converts a string like: "Action, Comedy, Fantasy"
    → {"action", "comedy", "fantasy"}

    Handles:
    - NaN values
    - Trailing spaces
    - Upper/lower-case automatically
    """

    if isinstance(genre_str, float) and np.isnan(genre_str):
        return set()

    return set(g.strip().lower() for g in str(genre_str).split(",") if g.strip() != "")


# -------------------------------------------------------------
# Build vocabulary and IDF vector for all genres
# -------------------------------------------------------------
def _build_genre_vocabulary(df: pd.DataFrame):
    """
    Creates:
    - vocab       : {"action": 0, "comedy": 1, ...}
    - idf_vector  : IDF score for each genre index

    IDF formula used:
        idf = log( (1 + N) / (1 + df) ) + 1
    """

    log_message("Building genre vocabulary...", "INFO")

    all_genre_sets = df["genre"].apply(_genre_to_set)
    vocab, doc_freq = {}, {}

    # Count how many anime contain each genre
    for genres in all_genre_sets:
        for g in genres:
            if g not in vocab:
                vocab[g] = len(vocab)
            doc_freq[g] = doc_freq.get(g, 0) + 1

    num_docs = len(df)
    num_genres = len(vocab)

    # Compute IDF for each genre
    idf_vector = np.zeros(num_genres, dtype=np.float32)
    for g, idx in vocab.items():
        df_g = doc_freq[g]
        idf_vector[idx] = np.log((1 + num_docs) / (1 + df_g)) + 1

    log_message(f"Genre vocabulary size: {num_genres}", "INFO")
    return vocab, idf_vector


# -------------------------------------------------------------
# Encode genres into a numerical TF-IDF-like matrix
# -------------------------------------------------------------
def _encode_genres(df: pd.DataFrame, vocab, idf_vector):
    """
    Builds a 2D NumPy array:
       shape = (num_anime, num_genres)

    Each anime row contains IDF scores for the genres it has.
    """

    log_message("Encoding genre vectors...", "INFO")

    num_rows = len(df)
    num_genres = len(vocab)
    genre_matrix = np.zeros((num_rows, num_genres), dtype=np.float32)

    for i, genre_str in enumerate(df["genre"]):
        genres = _genre_to_set(genre_str)
        for g in genres:
            idx = vocab.get(g)
            if idx is not None:
                genre_matrix[i, idx] = idf_vector[idx]

    return genre_matrix


# -------------------------------------------------------------
# Scale numeric features manually
# -------------------------------------------------------------
def _scale_numeric(df: pd.DataFrame, columns):
    """
    Converts numeric columns to:
         (value - mean) / std

    Returns:
    - Scaled numeric matrix
    - Stats (mean/std) for saving
    """

    log_message("Scaling numeric features...", "INFO")

    values = df[columns].to_numpy(dtype=np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0] = 1.0  # avoid divide-by-zero

    scaled = (values - mean) / std

    return scaled, {"mean": mean, "std": std}


# -------------------------------------------------------------
# MAIN FEATURE EXTRACTION FUNCTION
# -------------------------------------------------------------
def extract_features(df: pd.DataFrame):

    log_message("Starting Feature Engineering...", "INFO")
    t0 = start_timer()

    # 1. Build vocabulary + IDF
    vocab, idf_vector = _build_genre_vocabulary(df)

    # 2. Encode genres
    genre_matrix = _encode_genres(df, vocab, idf_vector)

    # 3. Scale numeric features
    numeric_cols = ["rating", "members", "episodes"]
    numeric_matrix, numeric_stats = _scale_numeric(df, numeric_cols)

    # 4. Combine genre + numeric
    feature_matrix = np.concatenate([genre_matrix, numeric_matrix], axis=1)

    # ---------------------------------------------------------
    # 5. Normalize feature vectors for cosine similarity
    # ---------------------------------------------------------
    """
    After normalization:
        dot(a, b) == cosine_similarity(a, b)
    So we avoid building a 12k × 12k matrix (saves RAM).
    """
    norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    feature_matrix = feature_matrix / norms

    log_message(f"Final feature matrix shape: {feature_matrix.shape}", "INFO")
    end_timer(t0, "Feature Engineering")

    # Save config so we can use it later if needed
    config = {
        "vocab": vocab,
        "idf_vector": idf_vector,
        "numeric_cols": numeric_cols,
        "numeric_mean": numeric_stats["mean"],
        "numeric_std": numeric_stats["std"],
    }

    return config, feature_matrix


# -------------------------------------------------------------
# Save artifacts
# -------------------------------------------------------------
def save_feature_artifacts(config, matrix,
                           config_path="models/feature_config.pkl",
                           matrix_path="data/processed/features_matrix.pkl"):

    ensure_folder(os.path.dirname(config_path))
    ensure_folder(os.path.dirname(matrix_path))

    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    with open(matrix_path, "wb") as f:
        pickle.dump(matrix, f)

    log_message("Feature artifacts saved.", "INFO")
