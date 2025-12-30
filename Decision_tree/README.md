# 🧠 Decision Tree Classification Pipeline

A complete **end-to-end Machine Learning pipeline** built using Python.

This project automates the full ML workflow including:
- ✔ Data Loading
- ✔ Preprocessing
- ✔ Feature Engineering
- ✔ Exploratory Data Analysis (EDA)
- ✔ Model Training
- ✔ Model Evaluation
- ✔ Visualizations
- ✔ Saving Outputs

This project is modular, scalable, professional, and production-ready.

---

## 📂 Project Structure

```
project/
│
├── data/
│   ├── raw/
│   │   └── heart_disease.csv
│   ├── processed/
│   │   └── cleaned_data.csv
│
├── models/
│   └── decision_tree_model.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── metrics.json
│   └── decision_tree_plot.png
│
├── reports/
│   ├── summary_statistics.csv
│   ├── missing_values.csv
│   ├── correlation_heatmap.png
│   └── feature_engineering_report.txt
│
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── train_model.py
│   ├── evaluate.py
│   ├── visualize_tree.py
│   └── main.py
│
├── create_dataset.py
└── README.md
```

---

## 🚀 Features

### 🔹 1. Automatic Folder Creation
All necessary project folders are auto-generated:
```
data/, data/raw/, data/processed/, models/, outputs/, reports/, logs/
```

---

### 🔹 2. Data Loading
- Loads dataset from `data/raw/heart_disease.csv`
- Validates file existence
- Displays shape & columns

---

### 🔹 3. Preprocessing
Includes:
- Duplicate removal
- Standardizing column names
- Missing value check
- Saving cleaned dataset

---

### 🔹 4. Feature Engineering
Includes:
- Label Encoding
- One-Hot Encoding
- Standard Scaling
- Generating a detailed feature engineering report

---

### 🔹 5. Exploratory Data Analysis (EDA)
Outputs saved in `reports/`:
- Summary statistics
- Missing value report
- Correlation heatmap
- Distribution plots for all numeric features

---

### 🔹 6. Model Training
Uses **DecisionTreeClassifier**

Automatically:
- Splits data
- Trains model
- Saves model to `/models` folder

---

### 🔹 7. Model Evaluation
Generates:
- Accuracy, Precision, Recall, F1
- Classification Report
- Confusion Matrix plot
- metrics.json

---

### 🔹 8. Decision Tree Visualization
- ✔ No GraphViz required
- ✔ Uses `sklearn.tree.plot_tree()`
- ✔ Saved as PNG in `outputs/decision_tree_plot.png`

---

## 📊 Example Metrics

```
ACCURACY: 0.9508
PRECISION: 0.9743
RECALL: 0.9508
F1_SCORE: 0.9580
```

---

## ▶️ How to Run the Pipeline

### Step 1 — Install requirements
```bash
pip install -r requirements.txt
```

### Step 2 — Generate the dataset (if not already generated)
```bash
python create_dataset.py
```

### Step 3 — Run the full ML pipeline
```bash
python src/main.py
```

All outputs will be saved automatically in their respective folders.

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

(Already included in `requirements.txt`)

---

## 🛠 Future Improvements

- Add Random Forest, XGBoost, SVM models
- Hyperparameter tuning (GridSearchCV)
- Build a Streamlit web app version
- Improve dataset quality
- Export model as ONNX / TensorFlow Lite

---

## ⭐ If you like this project, give it a star!

This motivates continued improvements and new ML projects.