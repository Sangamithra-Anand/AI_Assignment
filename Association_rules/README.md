# Market Basket Analysis with Apriori Algorithm

## 📌 Project Overview

This project performs Market Basket Analysis on a Groceries dataset using the Apriori algorithm. It automatically extracts:

* Frequent itemsets
* Association rules
* Strongest correlations
* Support, Confidence, Lift
* Visual plots and reports

**Pipeline flow:**

```
Load Data → Preprocess → Basket Encoding → Apriori → Rules → Reports + Visuals
```

---

## 📂 Project Structure

```
Association_Rules_Project/
│
│
├── data/
│   ├── raw/
│   │   └── groceries.csv
│   └── processed/
│       └── cleaned_groceries.csv
│
├── output/
│   ├── reports/
│   │   └── groceries_report.txt
│   ├── visuals/
│   │   ├── support_distribution.png
│   │   ├── confidence_distribution.png
│   │   ├── lift_scatter_plot.png
│   │   └── network_graph.png
│   ├── logs/
│   │   └── run_log.txt
│   ├── basket_format.csv
│   └── insights.txt
│
├── src/
│   ├── main.py
│   ├── load_data.py
│   ├── analyze_rules.py
│   ├── apriori_model.py
│   ├── generate_rules.py
│   ├── utils.py
│   ├── preprocess.py   (not used in this pipeline)
│   └── __init__.py
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run the Project

### 1️⃣ Install all required packages

```bash
pip install -r requirements.txt
```

📄 The `requirements.txt` file contains:

```
pandas
mlxtend
matplotlib
seaborn
networkx
openpyxl
numpy
```

### 2️⃣ Add your dataset

Place your dataset inside:

```
data/raw/
```

It must look like this:

```
milk,bread,eggs
shrimp,almonds,avocado
pasta,tomato sauce,cheese
```

### 3️⃣ Run the pipeline

```bash
python src/main.py
```

Everything else is automatic.

---

## 📊 Generated Outputs

### ✔ Visualizations (`output/visuals/`)

* `support_distribution.png`
* `confidence_distribution.png`
* `lift_scatter_plot.png`
* `network_graph.png`

### ✔ Text Insights (`output/insights.txt`)

Strongest rule, most confident rule.

### ✔ Full Analysis Report (`output/reports/groceries_report.txt`)

### ✔ Logs (`output/logs/run_log.txt`)

### ✔ Basket Format (`output/basket_format.csv`)

---

## 🧠 Why Market Basket Analysis?

It reveals hidden buying patterns, for example:

```
['milk', 'bread'] -> ['eggs']
Lift = 3.20
Confidence = 0.72
```

**Meaning:**

* Customers buying milk + bread are 3.2× more likely to also buy eggs.

**Useful for:**

* Recommendation systems
* Cross-selling
* Promotions
* Store layout optimization

---

## ⚙️ Technologies Used

* Python
* pandas
* mlxtend
* matplotlib
* seaborn
* networkx
* openpyxl

---

## 📝 License

This project is open source and available for educational and commercial use.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

