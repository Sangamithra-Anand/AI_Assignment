"""
tune_hyperparameters.py (FAST VERSION)
--------------------------------------
This file performs HYPERPARAMETER TUNING with:

- Progress bar + ETA
- Colored output
- Early stopping when accuracy ≥ threshold
- Summary of top models
- FAST hyperparameter grid (ONLY 2 COMBINATIONS)

This version does NOT use HYPERPARAM_GRID from config.py.
It contains its OWN INTERNAL FAST GRID so tuning always stays FAST.
"""

import os
import json
import time
from typing import Dict, Any, List

from dataclasses import replace
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm  # for progress bar with ETA

# Import helpers from train and config
from train import (
    load_processed_data,
    split_features_and_target,
    train_test_split_data,
    convert_labels_to_one_hot
)

from config import (
    TRAIN_PARAMS,
    MODELS_DIR,
    HYPERPARAM_RESULTS_PATH,
    TUNED_METRICS_PATH,
    create_directories
)

from model_builder import build_ann_model


# --------------------------------------------------------
# COLORS FOR BETTER VISUAL OUTPUT
# --------------------------------------------------------
def _color(t, c): return f"\033[{c}m{t}\033[0m"
def green(t): return _color(t, "32")
def yellow(t): return _color(t, "33")
def red(t): return _color(t, "31")
def cyan(t): return _color(t, "36")


# --------------------------------------------------------
# EARLY STOPPING — Stop tuning when val_acc >= threshold
# --------------------------------------------------------
EARLY_STOP_THRESHOLD = 0.93


# --------------------------------------------------------
# FAST HYPERPARAMETER GRID (ONLY 2 RUNS)
# --------------------------------------------------------
FAST_HYPERPARAM_GRID = {
    "hidden_layers": [1],
    "hidden_units": [32, 64],
    "activation": ["relu"],
    "learning_rate": [0.001],
    "batch_size": [32],
}


# --------------------------------------------------------
# Generate combinations manually
# --------------------------------------------------------
def generate_param_combinations(grid: Dict[str, List[Any]]):

    items = list(grid.items())
    if not items:
        return []

    def build(idx, current):
        if idx == len(items):
            return [current.copy()]

        name, values = items[idx]
        results = []
        for v in values:
            current[name] = v
            results.extend(build(idx + 1, current))
        return results

    return build(0, {})


# --------------------------------------------------------
# Run one experiment
# --------------------------------------------------------
def run_single_experiment(X_train, X_test, y_train, y_test, params):

    new_params = replace(
        TRAIN_PARAMS,
        hidden_layers=params["hidden_layers"],
        hidden_units=params["hidden_units"],
        activation=params["activation"],
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"]
    )

    print("\n" + cyan("======================================================"))
    print(cyan(f"[INFO] Running experiment with params:"))
    for k, v in params.items():
        print(cyan(f"  {k}: {v}"))
    print(cyan("======================================================\n"))

    input_dim = X_train.shape[1]
    num_classes = y_train.shape[1]

    model = build_ann_model(input_dim, num_classes, new_params)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=new_params.epochs,
        batch_size=new_params.batch_size,
        verbose=0
    )

    val_acc = float(history.history["val_accuracy"][-1])
    val_loss = float(history.history["val_loss"][-1])

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    return {
        "hidden_layers": params["hidden_layers"],
        "hidden_units": params["hidden_units"],
        "activation": params["activation"],
        "learning_rate": params["learning_rate"],
        "batch_size": params["batch_size"],
        "epochs": new_params.epochs,
        "final_val_accuracy": val_acc,
        "final_val_loss": val_loss,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss)
    }, model


# --------------------------------------------------------
# Save models + logs
# --------------------------------------------------------
def save_best_model(model, result):

    os.makedirs(MODELS_DIR, exist_ok=True)

    best_path = os.path.join(MODELS_DIR, "best_model.h5")
    print(f"[INFO] Saving best model to: {best_path}")
    model.save(best_path)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = os.path.join(MODELS_DIR, f"tuned_model_{ts}.h5")
    print(f"[INFO] Saving versioned tuned model to: {versioned_path}")
    model.save(versioned_path)

    os.makedirs(os.path.dirname(TUNED_METRICS_PATH), exist_ok=True)
    with open(TUNED_METRICS_PATH, "w") as f:
        json.dump(result, f, indent=4)


def save_summary(best_result, all_results):

    summary_path = os.path.join(os.path.dirname(HYPERPARAM_RESULTS_PATH), "best_hyperparameters.txt")

    sorted_results = sorted(all_results, key=lambda r: r["final_val_accuracy"], reverse=True)

    with open(summary_path, "w") as f:
        f.write("BEST HYPERPARAMETER SUMMARY\n")
        f.write("============================\n\n")
        f.write("Best parameters:\n")
        for k, v in best_result.items():
            f.write(f"  {k}: {v}\n")

        f.write("\nTop Configurations:\n")
        for idx, r in enumerate(sorted_results[:5], start=1):
            f.write(f"\nRank #{idx}\n")
            f.write(f"  val_acc = {r['final_val_accuracy']}\n")
            f.write(f"  test_acc = {r['test_accuracy']}\n")
            f.write(f"  params = {r}\n")

    print(f"[INFO] Saved hyperparameter summary to: {summary_path}")


# --------------------------------------------------------
# MAIN TUNING PIPELINE
# --------------------------------------------------------
def run_hyperparameter_search():

    df = load_processed_data()
    X, y_int = split_features_and_target(df)

    X_train, X_test, y_train_int, y_test_int = train_test_split_data(X, y_int)

    y_train, _ = convert_labels_to_one_hot(y_train_int)
    y_test, _ = convert_labels_to_one_hot(y_test_int)

    combos = generate_param_combinations(FAST_HYPERPARAM_GRID)

    print(f"[INFO] Total hyperparameter combinations to try: {len(combos)}")

    all_results = []
    best_model = None
    best_result = None
    best_acc = -1

    start = time.time()

    for params in tqdm(combos, desc="Tuning progress", unit="combo"):

        result, model = run_single_experiment(X_train, X_test, y_train, y_test, params)

        all_results.append(result)

        if result["final_val_accuracy"] > best_acc:
            best_acc = result["final_val_accuracy"]
            best_result = result
            best_model = model
            print(green(f"[INFO] New BEST accuracy = {best_acc:.4f}"))

            if best_acc >= EARLY_STOP_THRESHOLD:
                print(yellow("[EARLY STOP] Accuracy reached threshold. Stopping."))
                break

    print(f"[INFO] Tuning completed in {time.time() - start:.2f} seconds.")

    pd.DataFrame(all_results).to_csv(HYPERPARAM_RESULTS_PATH, index=False)

    save_best_model(best_model, best_result)
    save_summary(best_result, all_results)

    print(green("[INFO] Hyperparameter tuning finished successfully."))


# --------------------------------------------------------
if __name__ == "__main__":
    create_directories()
    run_hyperparameter_search()
