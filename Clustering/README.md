# Clustering Analysis Project

**K-Means | Hierarchical | DBSCAN | PCA Visualizations**

This project performs clustering analysis on the EastWestAirlines dataset using three major unsupervised learning algorithms:

- **K-Means Clustering**
- **Hierarchical (Agglomerative) Clustering**
- **DBSCAN (Density-Based Clustering)**

It also includes:
- Data preprocessing
- Outlier handling
- Feature scaling
- PCA visualizations
- Model evaluation using silhouette scores
- Automatic folder creation
- Automatic report generation

All modules are written in pure Python (no Jupyter notebooks) and follow a clean, scalable project structure.

---

## 📂 Project Structure

```
Clustering/
│
├── data/
│   ├── raw/
│   │    └── EastWestAirlines.xlsx
│   └── processed/
│        └── cleaned_data.csv
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── load_data.py
│   ├── preprocess.py
│   ├── kmeans_model.py
│   ├── hierarchical_model.py
│   ├── dbscan_model.py
│   ├── visualize.py
│   ├── evaluate.py
│   └── utils.py
│
├── output/
│   ├── plots/
│   ├── labels/
│   └── reports/
│
├── README.md
└── requirements.txt
```

---

## ▶️ How to Run the Project

### Step 1: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 2: Ensure Dataset Exists

Place your dataset here:

```
data/raw/EastWestAirlines.xlsx
```

### Step 3: Run the Project

```bash
python src/main.py
```

The script will automatically:
- Create output folders
- Detect correct Excel sheet
- Clean & scale data
- Run all clustering models
- Generate visualizations
- Save labels
- Produce a final report

---

## 🛠 What Each Module Does

### `load_data.py`
- Reads Excel file
- Auto-detects sheet with numeric data
- Returns raw + numeric DataFrames

### `preprocess.py`
- Removes duplicates
- Fills missing values
- Handles outliers using clipping
- Scales features
- Saves cleaned dataset

### `kmeans_model.py`
- Computes Elbow method
- Calculates silhouette scores
- Selects optimal K
- Saves cluster labels

### `hierarchical_model.py`
- Creates dendrogram
- Auto-selects best number of clusters
- Saves labels

### `dbscan_model.py`
- Generates K-distance plot
- Auto-selects eps
- Handles noise points
- Saves labels

### `visualize.py`
- Applies PCA
- Creates 2D cluster visualizations for all models

### `evaluate.py`
- Compares KMeans, Hierarchical, DBSCAN
- Summarizes silhouettes
- Reports number of clusters & noise
- Saves final analysis report

### `utils.py`
- Folder creation
- Safe plotting helpers
- Common utility functions

---

## 📊 Outputs Generated

After running the project, these files will be created automatically:

### 📁 output/plots/
- `elbow_plot.png`
- `dendrogram.png`
- `kmeans_pca_plot.png`
- `hierarchical_pca_plot.png`
- `dbscan_kdist.png`
- `dbscan_pca_plot.png`

### 📁 output/labels/
- `kmeans_labels.csv`
- `hierarchical_labels.csv`
- `dbscan_labels.csv`

### 📁 output/reports/
- `clustering_report.txt` (final summary of all models)

---

## 📈 Evaluation Metrics Used

### Silhouette Score
Used for:
- K-Means
- Hierarchical Clustering
- DBSCAN (excluding noise)

### Cluster Count
- **K-Means** → chosen K
- **Hierarchical** → chosen cluster count
- **DBSCAN** → clusters & noise

---

## 🧠 Key Concepts Learned

- Importance of preprocessing in clustering
- Outlier handling using percentile clipping
- Scaling using StandardScaler
- Dimensionality reduction via PCA
- Density-based clustering
- Choosing K using Elbow + Silhouette
- Automatic eps selection for DBSCAN
- Model comparison and evaluation

---

## 📦 Requirements

See `requirements.txt` for a complete list of dependencies. Key libraries include:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- scipy
- openpyxl

---

## 📝 License

This project is open-source and available for educational purposes.

---

## 👤 Author

**beencoder** - Created as part of a machine learning clustering analysis study.

---

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements or additional clustering algorithms.

---

## 📧 Contact

For questions or suggestions, please open an issue in the repository.