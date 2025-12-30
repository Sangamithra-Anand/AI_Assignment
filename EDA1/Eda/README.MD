# 📊 Automated Exploratory Data Analysis (EDA) Pipeline

This project is a **fully automated Exploratory Data Analysis (EDA) system** built using **Python** and designed to run inside **VS Code**.  

It loads a dataset, cleans it, analyzes it, generates statistical summaries, detects outliers, creates visualizations, and saves everything into a structured output folder.

This pipeline was built for the **Cardiotocographic (CTG) dataset**, but works with any CSV dataset.

---

## 🚀 Features

### ✔ Automated EDA Workflow

The pipeline performs:

- Dataset loading  
- Overview & structure analysis  
- Summary statistics  
- Missing value analysis  
- Missing value imputation (median)  
- Outlier detection using Z-score  
- Correlation analysis  
- Data distributions  
- Boxplots  
- Scatter plots for top correlated features  

---

## 📁 Project Structure

```
cardio-eda/
│
├── data/
│   └── Cardiotocographic.csv
│
├── src/
│   ├── config.py
│   ├── helpers.py
│   └── eda_main.py
│
├── outputs/
│   ├── figures/
│   └── summary/
│       ├── basic_stats.csv
│       ├── missing_report.csv
│       ├── correlation_matrix.csv
│       ├── overview.json
│       ├── outliers_indices.txt
│       └── data_cleaned.csv
│
├── requirements.txt
└── README.md
```

---

## 🛠 Technologies Used

- **Python 3.10+**
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- scipy  

---

## 🔧 Installation

### 1️⃣ Create a virtual environment

```bash
python -m venv .venv
```

### 2️⃣ Activate the environment (Windows)

```powershell
.venv\Scripts\Activate.ps1
```

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

### 3️⃣ Install required dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the EDA Pipeline

Run the following command in the project root:

```bash
python src/eda_main.py
```

When the script finishes, all outputs will be saved automatically in:

```
outputs/
├── figures/
└── summary/
```

---

## 📊 Output Files

### **Summary files (outputs/summary/):**

- `overview.json` — dataset shape, datatypes, missing values  
- `basic_stats.csv` — descriptive statistics  
- `missing_report.csv` — missing value analysis  
- `correlation_matrix.csv` — correlation table  
- `outliers_indices.txt` — detected outliers  
- `data_cleaned.csv` — cleaned dataset  

### **Figures (outputs/figures/):**

- `corr_heatmap.png`
- `dist_<column>.png`
- `box_<column>.png`
- `scatter_<x>_<y>.png`

---

## 📷 Example Screenshots

Add your visualization screenshots here:

```
outputs/
  figures/
    corr_heatmap.png
    dist_LB.png
    box_AC.png
```

---

## 📌 Customization

You can customize:

- Missing value strategy  
- Outlier threshold  
- Number of scatter plots  
- Graph style  
- Output folder names  

All editable inside `helpers.py` and `eda_main.py`.

