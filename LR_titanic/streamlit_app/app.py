# streamlit_app/app.py
"""
Streamlit app for Titanic Logistic Regression

Features:
- Single passenger prediction
- Batch CSV prediction
- Evaluation report display
- Feature importance visualization
- Histograms, Confusion Matrix
- Full NaN-safe preprocessing (fix for ValueError)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -------------------------
# PATH CONFIG
# -------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "output", "clean_train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic_model.pkl")
EVAL_REPORT_PATH = os.path.join(BASE_DIR, "output", "reports", "evaluation_report.txt")

# Same mappings as preprocessing.py
SEX_MAP = {"male": 0, "female": 1}
EMBARKED_MAP = {"S": 0, "C": 1, "Q": 2}

# -------------------------
# LOADING FUNCTIONS
# -------------------------
@st.cache_data
def load_clean_data():
    return pd.read_csv(CLEAN_DATA_PATH) if os.path.exists(CLEAN_DATA_PATH) else None

@st.cache_data
def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def get_training_stats(clean_df):
    """Extract medians/modes + feature order used in training."""
    if clean_df is None:
        return {}

    stats = {
        "Age_median": float(clean_df["Age"].median()),
        "Fare_median": float(clean_df["Fare"].median()),
        "Embarked_mode": clean_df["Embarked"].mode()[0],
        "feature_order": [c for c in clean_df.columns if c != "Survived"]
    }
    return stats

# -------------------------
# PREPROCESSING FUNCTIONS
# -------------------------
def preprocess_single_input(inputs, stats):
    """Convert one passenger's input into model-ready row."""

    row = {
        "Pclass": int(inputs["Pclass"]),
        "Sex": SEX_MAP[inputs["Sex"]],
        "Age": float(inputs["Age"]),
        "SibSp": int(inputs["SibSp"]),
        "Parch": int(inputs["Parch"]),
        "Fare": float(inputs["Fare"]),
        "Embarked": EMBARKED_MAP[inputs["Embarked"]],
        "FamilySize": int(inputs["SibSp"]) + int(inputs["Parch"]) + 1
    }

    # Create DF and match training feature order
    df_row = pd.DataFrame([row])
    for feature in stats["feature_order"]:
        if feature not in df_row.columns:
            df_row[feature] = 0
    df_row = df_row[stats["feature_order"]]

    # Final NaN cleanup (MOST IMPORTANT FIX)
    df_row = df_row.fillna(0)

    return df_row


def preprocess_batch(df, stats):
    """Preprocess uploaded CSV for batch prediction."""

    df = df.copy()

    # Fill missing
    df["Age"] = df.get("Age", stats["Age_median"]).fillna(stats["Age_median"])
    df["Fare"] = df.get("Fare", stats["Fare_median"]).fillna(stats["Fare_median"])

    # Encode Sex
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].astype(str).str.lower().map(SEX_MAP).fillna(0)
    else:
        df["Sex"] = 0

    # Encode Embarked
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].astype(str).map(EMBARKED_MAP).fillna(stats["Embarked_mode"])
    else:
        df["Embarked"] = stats["Embarked_mode"]

    # FamilySize
    df["FamilySize"] = df.get("SibSp", 0).fillna(0).astype(int) + df.get("Parch", 0).fillna(0).astype(int) + 1

    # Add missing columns & reorder
    for col in stats["feature_order"]:
        if col not in df.columns:
            df[col] = 0
    df = df[stats["feature_order"]]

    df = df.fillna(0)  # FINAL FIX — absolutely NO NaN allowed

    return df

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="Titanic Predictor", layout="wide")
st.title("🚢 Titanic Survival Prediction App (Logistic Regression)")

clean_df = load_clean_data()
model = load_model()
stats = get_training_stats(clean_df)

# Sidebar info
st.sidebar.header("Status")
if model:
    st.sidebar.success("Model loaded successfully.")
else:
    st.sidebar.error("Model not found. Train model first.")

if clean_df is not None:
    st.sidebar.info(f"Training Records: {len(clean_df)}")
else:
    st.sidebar.warning("Clean training data not found.")

# Evaluation report display
st.sidebar.header("Evaluation Report")
if os.path.exists(EVAL_REPORT_PATH):
    with open(EVAL_REPORT_PATH) as f:
        st.sidebar.text_area("Report", f.read(), height=200)

# ----------------------------------------------------
# SINGLE PREDICTION UI
# ----------------------------------------------------
st.subheader("🎯 Single Passenger Prediction")

col1, col2 = st.columns([2, 1])

with col1:
    with st.form("single_form"):
        Pclass = st.selectbox("Passenger Class", [1, 2, 3])
        Sex = st.radio("Sex", ["male", "female"])
        Age = st.number_input("Age", min_value=0.0, max_value=120.0, value=stats["Age_median"])
        SibSp = st.number_input("SibSp", 0, 10)
        Parch = st.number_input("Parch", 0, 10)
        Fare = st.number_input("Fare", min_value=0.0, value=stats["Fare_median"])
        Embarked = st.selectbox("Embarked", ["S", "C", "Q"])

        submit = st.form_submit_button("Predict")

    if submit:
        if model is None:
            st.error("No model. Train first.")
        else:
            # Build raw input
            raw = {
                "Pclass": Pclass, "Sex": Sex, "Age": Age,
                "SibSp": SibSp, "Parch": Parch,
                "Fare": Fare, "Embarked": Embarked
            }
            X_single = preprocess_single_input(raw, stats)

            # Predict
            prob = model.predict_proba(X_single)[0, 1]
            survived = prob >= 0.5

            st.metric("Survival Probability", f"{prob:.3f}")
            if survived:
                st.success("Prediction: SURVIVED")
            else:
                st.warning("Prediction: DID NOT SURVIVE")

# ----------------------------------------------------
# BATCH CSV UPLOAD
# ----------------------------------------------------
with col2:
    st.subheader("📦 Batch Predictions")
    up = st.file_uploader("Upload CSV", type=["csv"])

    if up:
        try:
            df_up = pd.read_csv(up)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            df_up = None

        if df_up is not None:
            st.write("Uploaded File Preview:")
            st.dataframe(df_up.head())

            X_batch = preprocess_batch(df_up, stats)

            # Safe predict — NO NANs after preprocess
            probs = model.predict_proba(X_batch)[:, 1]
            preds = (probs >= 0.5).astype(int)

            # Build results
            result_df = df_up.copy()
            result_df["survival_prob"] = probs
            result_df["predicted_survived"] = preds

            st.write("Prediction Results:")
            st.dataframe(result_df.head(10))

            # Download button
            buff = io.BytesIO()
            buff.write(result_df.to_csv(index=False).encode())
            buff.seek(0)
            st.download_button("Download predictions CSV", buff, "predictions.csv")

            # Histogram
            st.write("Probability Histogram:")
            fig, ax = plt.subplots()
            ax.hist(probs, bins=20)
            ax.set_xlabel("Survival Probability")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            # Confusion matrix if labels exist
            if "Survived" in df_up.columns:
                st.write("Confusion Matrix:")
                cm = confusion_matrix(df_up["Survived"], preds)
                fig2, ax2 = plt.subplots()
                disp = ConfusionMatrixDisplay(cm)
                disp.plot(ax=ax2)
                st.pyplot(fig2)

# ----------------------------------------------------
# FEATURE IMPORTANCE
# ----------------------------------------------------
st.markdown("---")
st.subheader("📊 Logistic Regression Feature Importance")

if model and stats:
    coefs = model.coef_.flatten()
    features = stats["feature_order"]

    feat_df = pd.DataFrame({
        "feature": features,
        "coef": coefs,
        "abs_coef": np.abs(coefs)
    }).sort_values("abs_coef", ascending=False)

    st.dataframe(feat_df[["feature", "coef"]].head(10))

    fig3, ax3 = plt.subplots(figsize=(8,5))
    ax3.barh(feat_df["feature"].head(10)[::-1], feat_df["coef"].head(10)[::-1])
    ax3.set_title("Top Feature Coefficients")
    st.pyplot(fig3)
else:
    st.info("Model not available.")

