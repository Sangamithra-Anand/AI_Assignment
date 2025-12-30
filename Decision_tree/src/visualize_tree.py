"""
visualize_tree.py
------------------
This file is responsible for VISUALIZING the trained Decision Tree.

What this script does:
✔ Creates a .png visualization of the decision tree
✔ Uses sklearn.tree.plot_tree() instead of GraphViz
✔ Saves the final tree image to outputs/decision_tree_plot.png
✔ Works directly with the model trained in train_model.py

All steps are explained inside the code.
"""

import os
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


def visualize_tree(model, feature_names, class_names=None,
                   output_file="outputs/decision_tree_plot.png"):
    """
    Create and save a visual representation of the Decision Tree.

    Parameters:
    -----------
    model : DecisionTreeClassifier
        Trained decision tree model.

    feature_names : list
        Names of the input features (X columns).

    class_names : list or None
        Names of target classes.
        If None → numeric labels are used.

    output_file : str
        Path to save the image file.

    Returns:
    --------
    None (saves PNG image of the decision tree)
    """

    print("\n[INFO] Starting Decision Tree visualization...")

    # ----------------------------------------------------------
    # 1. Validate the model
    # ----------------------------------------------------------
    if model is None:
        print("[ERROR] Cannot visualize — model is None.")
        return

    # ----------------------------------------------------------
    # 2. Ensure output folder exists
    # ----------------------------------------------------------
    os.makedirs("outputs", exist_ok=True)

    # ----------------------------------------------------------
    # 3. Plot the decision tree using sklearn (NO GraphViz needed)
    # ----------------------------------------------------------
    print("[INFO] Generating tree plot using sklearn.plot_tree()...")

    plt.figure(figsize=(25, 15))

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=[str(c) for c in class_names] if class_names else None,
        filled=True,
        rounded=True,
        fontsize=10
    )

    plt.title("Decision Tree Visualization")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Decision Tree image saved to: {output_file}")
    print("[INFO] Decision Tree visualization completed.\n")


# ============================================================
# TEST MODE: Run this file directly
# ============================================================
if __name__ == "__main__":
    print("[TEST] Running visualize_tree.py directly...")

    from load_data import load_raw_dataset
    from preprocess import preprocess_data
    from train_model import train_decision_tree
    from feature_engineering import feature_engineering

    raw_df = load_raw_dataset()

    if raw_df is not None:
        clean_df = preprocess_data(raw_df)

        # Apply feature engineering for test mode
        df_final = feature_engineering(
            clean_df,
            label_encode_cols=["sex", "fbs", "exang"],
            one_hot_cols=["cp", "restecg", "slope", "thal"],
            scale_cols=["age", "trestbps", "chol", "thalach", "oldpeak"]
        )

        TARGET = "num"

        model, X_test, y_test = train_decision_tree(df_final, TARGET)

        if model:
            feature_names = list(df_final.drop(columns=[TARGET]).columns)
            visualize_tree(model, feature_names)
        else:
            print("[TEST ERROR] Model training failed — cannot visualize.")
