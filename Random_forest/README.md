# 🧪 Glass Classification Using Random Forest, Bagging & Boosting

A complete end-to-end Machine Learning pipeline built in Python to classify glass types using the **UCI Glass Identification Dataset**.  
This project includes **EDA, data cleaning, preprocessing, visualizations, Random Forest model training, Bagging/Boosting, and detailed performance reports**.

---

## ⭐ Project Highlights

- ✔ Fully automated ML pipeline (no Jupyter Notebook required)  
- ✔ Auto-cleaning of messy Excel data (skips textual description rows)  
- ✔ Exploratory Data Analysis + automated EDA text report  
- ✔ Visualizations (histograms, boxplots, correlation heatmaps)  
- ✔ Preprocessing (duplicate removal, scaling, SMOTE oversampling)  
- ✔ Machine Learning:
  - 🌳 **Random Forest Classifier**
  - 🧺 **Bagging Classifier**
  - 🚀 **AdaBoost Classifier**
- ✔ Model saving (`.pkl` format)  
- ✔ Logging + time tracking  
- ✔ Fully modular & production-ready folder structure  

---

## 📂 Project Structure

```
Glass-RandomForest-Project/
│
├── data/
│   ├── raw/                    # (MANUAL) Place glass.xlsx here
│   │   └── glass.xlsx
│   └── processed/              # (AUTO) cleaned_glass.csv saved here
│
├── src/
│   ├── load_data.py            # Load & clean raw data
│   ├── eda.py                  # Exploratory Data Analysis
│   ├── visualize.py            # Histograms, boxplots, heatmaps
│   ├── preprocess.py           # Scaling, SMOTE, cleaning
│   ├── train_random_forest.py  # Train Random Forest model
│   ├── bagging_boosting.py     # Bagging & AdaBoost models
│   ├── evaluate.py             # Reusable evaluation module
│   ├── utils.py                # Logging, timers, folder handling
│   └── main.py                 # MASTER PIPELINE CONTROLLER
│
├── models/                     # (AUTO) Saved ML models
│
├── reports/                    # (AUTO) EDA + model performance reports
│
├── outputs/                    # (AUTO) Generated plots
│   ├── histograms/
│   ├── boxplots/
│   └── heatmaps/
│
├── logs/                       # (AUTO) Pipeline logs
│
├── requirements.txt            # Required Python packages
│
└── README.md                   # (THIS DOCUMENT)
```

---

## 📊 Dataset Description (UCI Glass Identification)

The dataset contains **chemical analysis of glass samples**, used for forensics.

| Feature | Description |
|---------|-------------|
| RI | Refractive Index |
| Na | Sodium |
| Mg | Magnesium |
| Al | Aluminum |
| Si | Silicon |
| K  | Potassium |
| Ca | Calcium |
| Ba | Barium |
| Fe | Iron |
| Type | Glass class label (1–7) |

**Classes include:**
- Building windows (float, non-float)
- Vehicle windows
- Containers
- Tableware
- Headlamps

---

## 🛠 Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Place your dataset here:**
```
data/raw/glass.xlsx
```

---

## ▶️ Running the Full Pipeline

Run this single command:

```bash
python src/main.py
```

**This performs:**

1. Load raw data  
2. Clean data  
3. EDA + text report  
4. Visualizations (saved in outputs/)  
5. Preprocessing (SMOTE + scaling)  
6. Random Forest training  
7. Bagging & Boosting training  
8. Save all models & reports  

---

## 📈 Model Performance Summary

### 🌳 Random Forest
```
Accuracy:  0.913
Precision: 0.913
Recall:    0.913
F1-score:  0.913
```

### 🧺 Bagging Classifier
```
Accuracy:  0.880
Precision: 0.877
Recall:    0.880
F1-score:  0.878
```

### 🚀 AdaBoost Classifier
```
Accuracy:  0.445
Precision: 0.366
Recall:    0.445
F1-score:  0.389
```

**Key Findings:**
- ➡ **Random Forest performed the best**  
- ➡ Bagging is reliable  
- ➡ AdaBoost performs poorly due to overlapping class boundaries

---

## 📊 Visual Outputs

Generated automatically inside `outputs/`:

- Histograms  
- Boxplots  
- Correlation heatmap  

These help understand feature distributions and relationships.

---

## 📝 Reports Generated

Inside `reports/` you get:

- `eda_report.txt`  
- `model_performance.txt`  
- `comparison_results.txt`  

Perfect for academic submission or project documentation.

---

## 🚀 Future Improvements

You can extend the project by adding:

- Gradient Boosting / XGBoost / LightGBM  
- Streamlit web app for live predictions  
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)  
- Confusion matrix visuals  
- Feature importance plots  

---



## 📄 License

This project is open-source and available under the MIT License.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---

## ⭐ Show your support

Give a ⭐️ if this project helped you!