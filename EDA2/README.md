# 📘 Adult Income Data Processing Pipeline

A complete end-to-end data preprocessing and feature engineering pipeline built in Python 3.13.6. This project processes the Adult Census dataset to prepare it for machine learning tasks such as income prediction.

## 🚀 Project Features

This pipeline performs:

### ✔ 1. Data Loading
Loads the dataset from `data/raw/adult_with_headers.csv`.

### ✔ 2. Preprocessing
* Handles missing values
* Encodes categorical columns
* Scales numerical features
* Saves cleaned data to `data/processed/cleaned_data.csv`

### ✔ 3. Feature Engineering
* Creates `age_group` (Young, Middle-Aged, Senior)
* Creates `capital_net` (capital_gain – capital_loss)
* Applies log transformation to skewed features
* Saves engineered dataset to `data/processed/engineered_data.csv`

### ✔ 4. Feature Selection
Using:
* **Isolation Forest** → detects and removes outliers
* **Mutual Information (MI)** → identifies which features are most predictive

Outputs:
* Outlier-free dataset → `output/outliers_removed.csv`
* MI feature importance → `output/mutual_information.csv`

### ✔ 5. Visualizations
Generates:
* Correlation Matrix Heatmap → `output/correlation_matrix.png`
* Mutual Information Heatmap → `output/mutual_information_heatmap.png`

### ✔ 6. Clear Modular Code
All logic is inside `src/`:

```
src/
│-- main.py
│-- load_data.py
│-- preprocess.py
│-- feature_engineering.py
│-- feature_selection.py
│-- visualization.py
│-- utils.py
│-- __init__.py
```

## 📁 Project Structure

```
EDA2/
│
├── data/
│   ├── raw/
│   │   └── adult_with_headers.csv
│   ├── processed/
│   │   ├── cleaned_data.csv
│   │   └── engineered_data.csv
│   └── reports/
│       └── eda_report.txt
│
├── output/
│   ├── outliers_removed.csv
│   ├── mutual_information.csv
│   ├── correlation_matrix.png
│   └── mutual_information_heatmap.png
│
├── src/
│   ├── main.py
│   ├── load_data.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── visualization.py
│   ├── utils.py
│   └── __init__.py
│
└── requirements.txt
```

## 🛠️ Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
scipy
```

All packages fully support Python 3.13.6.

## ▶️ How to Run the Project

Open terminal inside the project root folder and run:

```bash
python -m src.main
```

This will execute the entire pipeline step-by-step.

## 📊 Outputs Generated

After running the pipeline, you will get:

### 📄 Processed Data
* `cleaned_data.csv`
* `engineered_data.csv`
* `outliers_removed.csv`

### 📈 Visualizations
* `correlation_matrix.png`
* `mutual_information_heatmap.png`

### 🧠 Analysis
* `eda_report.txt`
* `mutual_information.csv`

## 📚 Key Techniques Used

### 🔹 Missing Value Handling
* Numerical → Median
* Categorical → Mode

### 🔹 Encoding
* One-Hot Encoding for small categories
* Label Encoding for large categories

### 🔹 Scaling
* StandardScaler → Normalization
* MinMaxScaler → 0–1 scaling

### 🔹 Outlier Detection
* Isolation Forest

### 🔹 Feature Selection
* Mutual Information (compatible with Python 3.13)

### 🔹 Feature Engineering
* Binning
* New numerical features
* Log transformation

## 🧩 Why This Pipeline Is Useful?

This project demonstrates how to build a real ML data preparation pipeline including:
* Clean architecture
* Modular code
* Scalable structure
* Reproducible workflow
* Visual + statistical feature selection

Perfect for Data Science projects, ML competitions, or academic assignments.

## ✅ Next Possible Extensions

You can expand this project by adding:
* Machine Learning Models (Logistic Regression, Random Forest, XGBoost)
* Streamlit Dashboard to visualize insights
* Automated model training pipeline
* Hyperparameter tuning

Ask if you'd like help with any of these!

**Happy Learning! 🚀**