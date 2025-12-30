# 🍷 PCA + K-Means Clustering Project  
Dimensionality Reduction & Clustering on the Wine Dataset

This project demonstrates a complete Machine Learning workflow using:

- **Exploratory Data Analysis (EDA)**
- **Data Preprocessing**
- **Principal Component Analysis (PCA)**
- **K-Means Clustering (Before & After PCA)**
- **Cluster Quality Evaluation**
- **Performance Comparison Between Original and PCA Data**

The project is fully automated through a single pipeline:

```bash
python src/main.py
```

---

## 📌 1. Project Overview

The goal of this project is to understand how **dimensionality reduction** using PCA affects clustering performance.  
We apply **K-Means** clustering twice:

1. **On Original Scaled Features**  
2. **On PCA-Transformed Features**

We then compare both results using:

- **Silhouette Score**
- **Davies–Bouldin Index**

A final comparison report is generated automatically.

---

## 🛠️ 2. Technologies Used

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-Learn  

---

## 📂 3. Project Folder Structure

```
pca_clustering_project/
│
├── data/
│   ├── raw/                          ← Manual
│   │   └── wine.csv
│   └── processed/                    ← Auto
│       ├── scaled_wine.csv
│       └── pca_transformed.csv
│
├── outputs/
│   ├── eda_plots/                    ← Auto
│   ├── pca_plots/                    ← Auto
│   ├── clustering_plots/             ← Auto
│   ├── visuals/                      ← Auto
│   └── reports/                      ← Auto
│       ├── pca_summary.json
│       ├── clustering_original_scores.json
│       ├── clustering_pca_scores.json
│       └── comparison_report.md
│
├── src/
│   ├── load_data.py
│   ├── eda.py
│   ├── preprocess.py
│   ├── pca_model.py
│   ├── clustering_original.py
│   ├── clustering_pca.py
│   ├── compare_results.py
│   ├── visualize.py
│   └── main.py
│
├── requirements.txt
└── README.md
```

---

## ▶️ 4. How to Run the Project

### **1. Install requirements**
```bash
pip install -r requirements.txt
```

### **2. Run the full pipeline**
```bash
python src/main.py
```

Everything will be generated automatically inside the **outputs/** folder.

---

## 🔍 5. Workflow Summary

### **Step 1 — EDA**
- Histograms  
- Boxplots  
- Correlation Heatmap  

Saved at: `outputs/eda_plots/`

### **Step 2 — Preprocessing**
- Keep numeric columns  
- Fill missing values  
- Standard scaling  

Saved at: `data/processed/scaled_wine.csv`

### **Step 3 — PCA**
- Scree Plot  
- Cumulative Variance Plot  
- PCA-transformed dataset  

Saved at:  
- `outputs/pca_plots/`  
- `data/processed/pca_transformed.csv`

### **Step 4 — K-Means Clustering**
Clustering performed on:

- **Original Scaled Data**
- **PCA-Transformed Data**

Saved at:  
- `outputs/clustering_plots/`  
- `outputs/reports/`

### **Step 5 — Comparison Report**
Automatically generated evaluation showing which approach performed better.

Saved at:  
`outputs/reports/comparison_report.md`

---

## 🧠 6. Concepts Used

### **📌 Principal Component Analysis (PCA)**
- Reduces dimensionality  
- Removes correlation between features  
- Helps visualize high-dimensional data  
- Often improves clustering  

### **📌 K-Means Clustering**
- Unsupervised learning  
- Groups similar data points  
- Improved by scaling  
- Can perform better on PCA-transformed data  

### **📌 Evaluation Metrics**

| Metric | Meaning | Good Value |
|--------|---------|------------|
| **Silhouette Score** | Cluster separation quality | Closer to +1 |
| **Davies–Bouldin Index** | Cluster compactness | Lower is better |

---

## 📊 7. Key Outputs

| Output | Location | Description |
|--------|----------|------------|
| Histograms | `outputs/eda_plots/` | Distribution of each feature |
| Scree Plot | `outputs/pca_plots/` | Variance explained by components |
| PCA Dataset | `data/processed/pca_transformed.csv` | Reduced features |
| Cluster Plots | `outputs/clustering_plots/` | Visual cluster separation |
| Comparison Report | `outputs/reports/comparison_report.md` | Final evaluation |

---

## 🏁 8. Conclusion & Insights

- PCA helps simplify the dataset while keeping important information.  
- K-Means sometimes performs better on PCA-transformed data because:
  - Noise is reduced  
  - Highly correlated features are removed  
  - Lower dimensions → easier clustering  
- The comparison metrics reveal whether PCA improved clustering performance for this dataset.

This project demonstrates a **complete, end-to-end unsupervised learning workflow** suitable for real-world machine learning pipelines.

---

## 📞 Contact

Feel free to reach out if you have any questions or suggestions about this project!

