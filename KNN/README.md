# 🦁 KNN Zoo Classification Project  
A complete Machine Learning pipeline using **K-Nearest Neighbors (KNN)** to classify animals based on biological features.  
This project includes **data loading, preprocessing, model training, evaluation, and visualizations** — all implemented using Python.

---

## 📌 Project Overview

The goal of this project is to classify animals into their respective types (mammal, bird, fish, reptile, etc.) using the **Zoo dataset**.  
The dataset contains multiple features such as:

- hair  
- feathers  
- eggs  
- milk  
- airborne  
- aquatic  
- predator  
- backbone  
- legs  
- tail  
- domestic  
- catsize  
- type (target variable)

We use **KNN classification** to build and evaluate the model.

---

## 📁 Project Structure

```
KNN/
│
├── data/
│   └── Zoo.csv
│
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── visualize.py
│   ├── knn_model.py
│   ├── evaluate.py
│   └── main.py
│
├── output/              # Auto-created by program
│   ├── models/          # Saved KNN model
│   ├── plots/           # EDA & decision boundary plots
│   └── reports/         # Evaluation reports
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Full Pipeline

Run the main script:

```bash
python src/main.py
```

The pipeline performs:

1. Load dataset
2. Preprocess data
   - Missing value check
   - Outlier detection
   - Feature scaling
3. Find best K value for KNN
4. Train KNN model
5. Evaluate performance
6. Generate visualizations

All results are saved automatically in the `output/` folder.

---

## 📊 Output Files

### 🔹 Models (`output/models/`)
- `knn_model.pkl` → saved trained model

### 🔹 Reports (`output/reports/`)
- `classification_report.txt`
- `metrics.txt`
- `confusion_matrix.png`

### 🔹 Plots (`output/plots/`)
- Distribution plots for numeric features
- Correlation heatmap
- Decision boundary plot

---

## 🧠 Machine Learning Details

### Why KNN?
- Simple, intuitive algorithm
- Great for small- to medium-sized datasets
- Works well when the relationship between variables is non-linear

### Hyperparameters used
- **K (number of neighbors)** → chosen using accuracy-based tuning
- **Distance metric:**
  - Minkowski (default in sklearn, equivalent to Euclidean when p=2)

### Feature Scaling
Scaling is essential because KNN is distance-based. We use `StandardScaler` to normalize each feature:

```
new_value = (value – mean) / standard deviation
```

---

## 📈 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

These values are printed in the console and saved in the reports folder.

---

## 🧪 Visualizations

The project generates:

### ✔ Distribution Plots
Shows how each numeric feature is distributed.

### ✔ Correlation Heatmap
Uses only numeric columns to avoid errors.

### ✔ Decision Boundary Plot
Shows how KNN separates classes using two selected features.

---

## 📚 File Descriptions (src/)

### `load_data.py`
Loads `Zoo.csv` and creates output folders automatically.

### `preprocess.py`
Handles:
- Missing value check
- Outlier detection
- Feature scaling
- Ignores non-numeric columns like "animal name"

### `knn_model.py`
Handles:
- K-value tuning
- Training KNN model
- Saving trained model

### `evaluate.py`
Produces:
- Classification report
- Metrics summary
- Confusion matrix plot

### `visualize.py`
Generates:
- Histograms
- Correlation heatmap
- Decision boundary plot

### `main.py`
Runs the entire pipeline end-to-end.

---

## 🏁 Final Results

After running the full pipeline:

- Best K value is automatically found
- Model achieves high accuracy
- All visualizations and evaluation files are generated
- Project becomes fully reproducible and submission-ready