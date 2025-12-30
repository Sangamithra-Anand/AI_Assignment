# 📘 Toyota Corolla – Multiple Linear Regression (MLR) Project

A complete end-to-end Machine Learning Pipeline built using Python. This project analyzes the Toyota Corolla dataset and builds several regression models to predict car prices using numerical and categorical features.

---

## ✨ Project Features

### ✔ Exploratory Data Analysis (EDA)
- Summary statistics
- Distribution plots (histograms)
- Boxplots for outlier detection
- Correlation heatmap

### ✔ Data Preprocessing
- Remove duplicates
- Handle missing values
- Clean the `Doors` column
- One-Hot Encode categorical variables (`Fuel_Type`)
- Final cleaned dataset saved

### ✔ Model Training
Three Multiple Linear Regression models:
1. **Basic Linear Regression**
2. **Reduced Model** (after removing high-VIF multicollinear features)
3. **Scaled Model** (StandardScaler + Linear Regression)

### ✔ Model Evaluation
- MSE, RMSE
- MAE
- R² Score
- Comparison report saved to `/output/evaluation_results.txt`

### ✔ Regularization Models
- LassoCV
- RidgeCV
- Automatic hyperparameter tuning (cross-validation)
- Coefficient analysis report saved

### ✔ Menu-driven Interface (CLI)
User can:
- Run EDA
- Run preprocessing
- Train models
- Evaluate models
- Run regularization
- Or run the **FULL PIPELINE (RECOMMENDED)**

### ✔ Automatic folder creation
All required folders are created automatically.

---

## 📂 Project Structure

```
MLR/
│
├── data/
│   ├── raw/
│   │    └── ToyotaCorolla - MLR.csv
│   └── processed/
│        └── cleaned_data.csv
│
├── models/
│   ├── model_1_basic.pkl
│   ├── model_2_reduced.pkl
│   ├── model_3_scaled.pkl
│   ├── lasso_model.pkl
│   ├── ridge_model.pkl
│   ├── scaler.pkl
│   └── regularization_scaler.pkl
│
├── output/
│   ├── plots/
│   ├── eda_report.txt
│   ├── coefficient_summary.txt
│   ├── evaluation_results.txt
│   └── regularization_summary.txt
│
├── src/
│   ├── main.py
│   ├── eda.py
│   ├── preprocess.py
│   ├── train_models.py
│   ├── evaluate.py
│   ├── regularization.py
│   ├── utils.py
│   └── __init__.py
│
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Project

```bash
python src/main.py
```

You will see the interactive menu:

```
1. Run EDA
2. Run Preprocessing
3. Train Regression Models
4. Evaluate Models
5. Run Lasso & Ridge Regularization
6. Run FULL PIPELINE (RECOMMENDED)
7. Exit
```

**For most users, choose option 6.**

---

## 📈 Outputs Generated

### 🔹 EDA Outputs
Located in `output/plots/`
- Histograms
- Boxplots
- Correlation heatmap
- Summary report

### 🔹 Model Outputs
Located in `models/`
- Basic model
- Reduced model
- Scaled model
- Lasso model
- Ridge model

### 🔹 Evaluation Outputs
Located in `output/`
- Evaluation results
- Coefficient summary
- Regularization summary

---

## 🧠 Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **Statsmodels**
- **Seaborn**

---

## 📝 Notes

- The `Doors` column is cleaned using regex to remove non-numeric strings.
- VIF is used to detect multicollinearity; high-VIF features are removed.
- One-Hot Encoding avoids the dummy variable trap (`drop_first=True`).
- All models are saved in `.pkl` format for reuse.

---

## 📄 License

This project is open-source and available for educational purposes.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

## ⭐ Show your support

Give a ⭐️ if this project helped you!