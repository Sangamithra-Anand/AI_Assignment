# ---------------------------------------------------------------
# SVM TRAINING & EVALUATION MODULE
# ---------------------------------------------------------------
# This file contains functions to:
#   1. Train an SVM model
#   2. Predict on test data
#   3. Evaluate the model (accuracy, confusion matrix, classification report)
#   4. Save the trained model for future use
#
# NOTE: This module expects the data to be:
#   - Already encoded (numeric)
#   - Already split into X_train, X_test, y_train, y_test
#
# OUTPUT:
#   - Saves the model in models/svm_model.pkl
#   - Saves evaluation reports into output/reports/
# ---------------------------------------------------------------

import os
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ---------------------------------------------------------------
# FILE PATHS FOR SAVING OUTPUTS
# ---------------------------------------------------------------

# PROJECT_ROOT = the main folder of your project
# Example: D:/Projects/Data-Science/Work/SVM/mushroom_svm_project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Correct internal folders (NO going outside project)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
REPORT_DIR = os.path.join(PROJECT_ROOT, "output", "reports")

# Create folders if not available
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ---------------------------------------------------------------
# FUNCTION: train_svm_model()
# ---------------------------------------------------------------
# PURPOSE:
#   Train an SVM classifier on the training dataset.
#
# PARAMETERS:
#   X_train — training features
#   y_train — training labels
#   kernel  — which SVM kernel to use (default: 'rbf')
#
# RETURNS:
#   The trained SVC model object
# ---------------------------------------------------------------
def train_svm_model(X_train, y_train, kernel='rbf'):
    print(f"\nTraining SVM model with kernel = '{kernel}' ...")

    # Create the SVM classifier
    # C=1.0 is the regularization parameter
    model = SVC(kernel=kernel, C=1.0)

    # Fit (train) the model
    model.fit(X_train, y_train)

    print("Model training completed.")
    return model


# ---------------------------------------------------------------
# FUNCTION: evaluate_model()
# ---------------------------------------------------------------
# PURPOSE:
#   Evaluate the trained model using the test dataset.
#
# WHAT IT PRINTS:
#   - Accuracy score
#   - Confusion matrix
#   - Classification report (precision, recall, f1-score)
#
# WHAT IT SAVES:
#   A text file of the evaluation inside output/reports/
# ---------------------------------------------------------------
def evaluate_model(model, X_test, y_test, report_name="svm_evaluation.txt"):
    print("\nEvaluating SVM model on test data...")

    # Predict labels for test features
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nACCURACY: {accuracy:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nCONFUSION MATRIX:")
    print(cm)

    # Classification report
    report = classification_report(y_test, y_pred)
    print("\nCLASSIFICATION REPORT:")
    print(report)

    # Save results to a text file
    report_path = os.path.join(REPORT_DIR, report_name)
    with open(report_path, "w") as f:
        f.write(f"ACCURACY: {accuracy:.4f}\n\n")
        f.write("CONFUSION MATRIX:\n")
        f.write(str(cm) + "\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write(report)

    print(f"\nSaved evaluation report: {report_path}")

    return accuracy, cm, report


# ---------------------------------------------------------------
# FUNCTION: save_svm_model()
# ---------------------------------------------------------------
# PURPOSE:
#   Save the trained SVM model as a .pkl file
#   (pickle = Python format used for model storage)
#
# OUTPUT:
#   File saved inside models/svm_model.pkl
# ---------------------------------------------------------------
def save_svm_model(model, filename="svm_model.pkl"):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)

    print(f"\nModel saved successfully: {path}")

