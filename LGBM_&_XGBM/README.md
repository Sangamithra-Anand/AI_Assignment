# 📘 Titanic ML Pipeline — EDA → Preprocessing → LightGBM & XGBoost

## 🌟 Introduction

This project implements a complete, automated Machine Learning pipeline on the classic Titanic Survival Prediction dataset. The pipeline covers every stage of a real-world ML workflow:

✔ Data loading  
✔ Exploratory Data Analysis (EDA)  
✔ Preprocessing  
✔ Model training (LightGBM & XGBoost)  
✔ Model evaluation & comparison  
✔ Saving models, graphs, and reports  
✔ Fully automated using a single command

The goal is to provide a clean, modular, and industry-style ML architecture suitable for:

* Learning Machine Learning
* Demonstrating portfolio projects
* Kaggle-style competitions
* Academic or research submissions

---

## 🧩 Features

### 🔍 EDA (Exploratory Data Analysis)

The pipeline automatically generates visual insights:

* Age distribution
* Fare distribution
* Survival count
* Survival by gender
* Fare by passenger class

All plots are saved in:

```
output/graphs/
```

### 🧼 Data Preprocessing

The preprocessing pipeline handles:

* Missing values (Age, Fare, Embarked)
* Dropping high-missing features (Cabin)
* Encoding categorical variables (Sex, Embarked)
* Selecting important features
* Splitting training data into Train / Validation

### ⚡ Model Training

Two powerful gradient boosting models are trained:

* **LightGBM**
* **XGBoost**

Their models are saved as `.pkl` files:

```
models/lgbm_model.pkl
models/xgb_model.pkl
```

### 📊 Model Evaluation

Evaluation includes:

* Accuracy
* Classification report
* Confusion matrix
* Side-by-side model comparison

Outputs are saved in:

```
output/reports/
    ├── lightgbm_evaluation.txt
    ├── xgboost_evaluation.txt
    └── comparison_report.txt
```

### 🧠 Fully Automated Pipeline

Run everything (EDA → Preprocessing → Training → Evaluation) using:

```bash
python src/main.py
```

The script ensures all folders exist and executes every step sequentially.

---

## 📂 Project Structure

```
LGBM_&_XGBM/
│
├── data/
│   ├── Titanic_train.csv
│   └── Titanic_test.csv
│
├── models/                  # Saved models
│
├── output/
│   ├── graphs/              # EDA plots
│   └── reports/             # Evaluation reports
│
├── src/
│   ├── main.py              # Automated pipeline controller
│   ├── eda.py               # EDA + graph generation
│   ├── preprocess.py        # Data cleaning + encoding
│   ├── train_models.py      # LightGBM & XGBoost training
│   └── evaluate.py          # Model evaluation + comparison
│
├── requirements.txt
└── README.md
```

---

## 🚀 Running the Project

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run the full pipeline

```bash
python src/main.py
```

Everything will be generated automatically:

* Graphs → `output/graphs/`
* Reports → `output/reports/`
* Models → `models/`

---

## 🔧 Requirements

Main libraries:

* Python 3.10+
* pandas
* numpy
* seaborn
* matplotlib
* scikit-learn
* lightgbm
* xgboost
* joblib

All dependencies are included in `requirements.txt`.

---

## 🎯 Purpose of the Project

This project is designed to be:

* **Beginner-friendly**
* **Industry-style**
* **Easy to maintain**
* **Clear and modular**
* **Useful for learning ML pipeline design**

It can be extended with:

* Hyperparameter tuning
* Cross-validation
* Additional models
* Deployment (Flask, FastAPI, Streamlit)

Just ask if you want any of these upgrades!

---

## 🤝 Contributing

Feel free to fork the project and improve:

* Feature engineering
* Visualizations
* Model performance
* Documentation

Pull requests are welcome.

---

## 📄 License

This project is open-source and free to use for learning and development purposes.

---

**Happy Learning! 🚀**