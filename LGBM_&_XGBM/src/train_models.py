"""
TRAINING SCRIPT
----------------
This trains two ML models:

1. LightGBM
2. XGBoost

Both models are saved into:
    project_root/models/
"""

import os
import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score


# ------------------------------------------------------------
# TRAIN LIGHTGBM
# ------------------------------------------------------------
def train_lightgbm(X_train, y_train, X_val, y_val):

    print("\nTraining LightGBM...")

    # Model with simple parameters
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Validation accuracy
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print("LightGBM Accuracy:", round(acc, 4))

    return model


# ------------------------------------------------------------
# TRAIN XGBOOST
# ------------------------------------------------------------
def train_xgboost(X_train, y_train, X_val, y_val):

    print("\nTraining XGBoost...")

    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="binary:logistic"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print("XGBoost Accuracy:", round(acc, 4))

    return model


# ------------------------------------------------------------
# TRAIN BOTH MODELS + SAVE THEM
# ------------------------------------------------------------
def train_models(X_train, y_train, X_val, y_val):

    # Create the models folder
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    print("\nModel folder ensured:", model_dir)

    # Train LightGBM
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)
    joblib.dump(lgb_model, os.path.join(model_dir, "lgbm_model.pkl"))

    # Train XGBoost
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    joblib.dump(xgb_model, os.path.join(model_dir, "xgb_model.pkl"))

    print("\nTraining Completed!")
    return lgb_model, xgb_model


