"""
visualize_results.py
--------------------
This file is used to CREATE VISUAL PLOTS for:

1. Training loss curve (loss vs epochs)
2. Training accuracy curve (accuracy vs epochs)
3. Confusion matrix (for baseline and tuned models)

It reads:
- training_log.txt  -> created by train.py
- baseline_metrics.json / tuned_metrics.json -> created by train.py + evaluate.py

And saves:
- loss_curve.png           -> in output/figures/
- accuracy_curve.png       -> in output/figures/
- confusion_matrix.png     -> in output/figures/ (for one model, e.g., tuned)
"""

import os
import re
import json
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt

# Import paths from config.py
from config import (
    TRAINING_LOG_PATH,        # text log with loss/accuracy per epoch
    LOSS_CURVE_PATH,          # where to save loss curve image
    ACCURACY_CURVE_PATH,      # where to save accuracy curve image
    BASELINE_METRICS_PATH,    # JSON file for baseline metrics
    TUNED_METRICS_PATH,       # JSON file for tuned metrics
    CONFUSION_MATRIX_PATH,    # where to save confusion matrix image
    create_directories        # ensures folders exist
)


def parse_training_log(log_path: str) -> Dict[str, List[float]]:
    """
    Parse the training_log.txt file created by train.py and extract:

    - loss (training loss per epoch)
    - acc (training accuracy per epoch)
    - val_loss (validation loss per epoch)
    - val_acc (validation accuracy per epoch)

    Expected line format in training_log.txt (from train.py):
        Epoch 1: loss=0.6931, acc=0.5000, val_loss=0.6920, val_acc=0.5200

    We use regular expressions to extract the numbers.

    Args:
        log_path: path to the training log text file.

    Returns:
        A dictionary with keys:
            "loss", "accuracy", "val_loss", "val_accuracy"
        Each value is a list of floats (one per epoch).
    """
    if not os.path.exists(log_path):
        print(f"[WARNING] Training log not found at: {log_path}")
        return {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    # Initialize empty lists for each metric
    loss_list = []
    acc_list = []
    val_loss_list = []
    val_acc_list = []

    # Regular expression pattern to capture 4 floating point numbers
    # Example line:
    #   Epoch 1: loss=0.6931, acc=0.5000, val_loss=0.6920, val_acc=0.5200
    line_pattern = re.compile(
        r"loss=([0-9\.]+), acc=([0-9\.]+), val_loss=([0-9\.]+), val_acc=([0-9\.]+)"
    )

    print(f"[INFO] Parsing training log from: {log_path}")
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = line_pattern.search(line)
            if match:
                # Extract the 4 captured groups as floats
                loss = float(match.group(1))
                acc = float(match.group(2))
                val_loss = float(match.group(3))
                val_acc = float(match.group(4))

                # Append to respective lists
                loss_list.append(loss)
                acc_list.append(acc)
                val_loss_list.append(val_loss)
                val_acc_list.append(val_acc)

    print(f"[INFO] Parsed {len(loss_list)} epochs from training log.")

    return {
        "loss": loss_list,
        "accuracy": acc_list,
        "val_loss": val_loss_list,
        "val_accuracy": val_acc_list,
    }


def plot_loss_curve(history: Dict[str, List[float]], save_path: str) -> None:
    """
    Plot training and validation loss vs epochs.

    Args:
        history: dictionary with keys "loss" and "val_loss"
        save_path: file path where the plot image will be saved
    """
    if not history["loss"]:
        print("[WARNING] No loss history found. Skipping loss curve plot.")
        return

    epochs = range(1, len(history["loss"]) + 1)

    plt.figure(figsize=(8, 5))  # size of the figure in inches

    # Plot training loss
    plt.plot(epochs, history["loss"], label="Training Loss")

    # Plot validation loss (if exists)
    if history["val_loss"]:
        plt.plot(epochs, history["val_loss"], label="Validation Loss")

    plt.title("Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"[INFO] Saved loss curve to: {save_path}")


def plot_accuracy_curve(history: Dict[str, List[float]], save_path: str) -> None:
    """
    Plot training and validation accuracy vs epochs.

    Args:
        history: dictionary with keys "accuracy" and "val_accuracy"
        save_path: file path where the plot image will be saved
    """
    if not history["accuracy"]:
        print("[WARNING] No accuracy history found. Skipping accuracy curve plot.")
        return

    epochs = range(1, len(history["accuracy"]) + 1)

    plt.figure(figsize=(8, 5))

    # Plot training accuracy
    plt.plot(epochs, history["accuracy"], label="Training Accuracy")

    # Plot validation accuracy (if exists)
    if history["val_accuracy"]:
        plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")

    plt.title("Accuracy vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"[INFO] Saved accuracy curve to: {save_path}")


def load_confusion_matrix_from_metrics(metrics_path: str) -> Any:
    """
    Load confusion matrix from a metrics JSON file.

    Expected JSON structure (from evaluate.py):
        {
          "test_loss": ...,
          "test_accuracy": ...,
          "classification_metrics": {
            "overall": {...},
            "per_class": {...},
            "confusion_matrix": [[...], [...], ...]
          }
        }

    Args:
        metrics_path: path to the metrics JSON file.

    Returns:
        cm (2D list or np.ndarray) if available, otherwise None.
    """
    if not os.path.exists(metrics_path):
        print(f"[WARNING] Metrics file not found at: {metrics_path}")
        return None

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cm = None
    # Safely navigate through nested keys
    if "classification_metrics" in data:
        cm = data["classification_metrics"].get("confusion_matrix", None)

    if cm is None:
        print(f"[WARNING] No confusion matrix stored in: {metrics_path}")

    return np.array(cm) if cm is not None else None


def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: str) -> None:
    """
    Plot a confusion matrix as a heatmap.

    Args:
        cm: confusion matrix as a 2D numpy array
        title: title for the plot
        save_path: path to save the plotted image
    """
    if cm is None or cm.size == 0:
        print("[WARNING] Empty confusion matrix. Skipping confusion matrix plot.")
        return

    plt.figure(figsize=(7, 6))

    # Show matrix as an image
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    # Add axis labels
    num_classes = cm.shape[0]
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, tick_marks)  # class indices on X-axis
    plt.yticks(tick_marks, tick_marks)  # class indices on Y-axis

    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    # Write the numbers inside each cell
    thresh = cm.max() / 2.0  # threshold for text color
    for i in range(num_classes):
        for j in range(num_classes):
            value = cm[i, j]
            plt.text(
                j,
                i,
                str(value),
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if value > thresh else "black",
            )

    plt.tight_layout()

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    print(f"[INFO] Saved confusion matrix plot to: {save_path}")


def main_visualization_pipeline() -> None:
    """
    Full pipeline to create visualization plots.

    Steps:
    1. Parse training_log.txt and plot:
        - loss_curve.png
        - accuracy_curve.png
    2. Load confusion matrix from tuned_metrics.json (preferred)
       or fallback to baseline_metrics.json, and plot:
        - confusion_matrix.png
    """
    # -------------------------------
    # 1. Loss & Accuracy curves
    # -------------------------------
    history = parse_training_log(TRAINING_LOG_PATH)

    # Plot loss curve
    plot_loss_curve(history, LOSS_CURVE_PATH)

    # Plot accuracy curve
    plot_accuracy_curve(history, ACCURACY_CURVE_PATH)

    # -------------------------------
    # 2. Confusion Matrix
    # -------------------------------
    # Priority: use tuned model confusion matrix if available
    cm = load_confusion_matrix_from_metrics(TUNED_METRICS_PATH)

    used_source = "Tuned model"
    if cm is None:
        # Fallback to baseline if tuned metrics not available
        cm = load_confusion_matrix_from_metrics(BASELINE_METRICS_PATH)
        used_source = "Baseline model"

    if cm is not None:
        title = f"Confusion Matrix ({used_source})"
        plot_confusion_matrix(cm, title, CONFUSION_MATRIX_PATH)
    else:
        print("[WARNING] No confusion matrix found for either tuned or baseline model.")


# ----------------------------------------------------------------------
# Entry point: run visualization from terminal.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    """
    If you run:

        python visualize_results.py

    This will:
    1. Make sure required folders (output/figures, etc.) exist.
    2. Try to plot:
        - Loss curve      -> output/figures/loss_curve.png
        - Accuracy curve  -> output/figures/accuracy_curve.png
        - Confusion matrix -> output/figures/confusion_matrix.png
    """

    create_directories()
    main_visualization_pipeline()


