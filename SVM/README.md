# 🍄 Mushroom Classification using Support Vector Machine (SVM)

This project builds a complete machine learning pipeline to classify mushrooms as **edible** or **poisonous** using a **Support Vector Machine (SVM)** classifier.

It includes:

- Exploratory Data Analysis (EDA)
- Categorical Data Visualization
- Data Preprocessing
- SVM Model Training
- Model Evaluation
- Saving Output Files and Model
- Fully Modular Code Structure

---

## 📁 Project Structure

```
SVM/
│
├── data/
│   └── mushroom.csv           # Dataset file
│
├── models/
│   └── svm_model.pkl          # Saved trained model
│
├── output/
│   ├── graphs/                # Saved plots
│   └── reports/               # Model evaluation reports
│
├── src/
│   ├── main.py                # Main controller script
│   ├── eda.py                 # Loading + basic dataset inspection
│   ├── preprocess.py          # Encoding + train-test split
│   ├── visualize.py           # All visualization functions
│   └── train_svm.py           # Train, evaluate, and save SVM
│
└── requirements.txt           # Project dependencies
```

---

## 🚀 How to Run the Project

### **1. Go to project directory**
```bash
cd SVM
```

### **2. Run the project using Python module format**
```bash
python -m src.main
```

⚠️ **Important:** Do NOT run `python src/main.py`  
Use `python -m src.main` to ensure all imports and paths work correctly.

---

## 🧩 What the Project Does

### 1. Load Dataset
Loads `mushroom.csv` from the `data/` folder using `load_data()`.

### 2. Exploratory Data Analysis
- First 5 rows
- Dataset shape
- Info (column types)
- Missing values
- Class distribution

### 3. Visualizations
Saved inside `output/graphs/`:
- Class distribution plot
- Odor vs Class
- Habitat vs Class
- Correlation matrix
- Top correlated features

### 4. Preprocessing
- Encodes all categorical columns → numbers
- Splits dataset: 80% training, 20% testing

### 5. SVM Model Training
Uses RBF kernel, best for mushroom dataset.

### 6. Evaluation
Saves in: `output/reports/svm_evaluation.txt`

Includes:
- Accuracy
- Confusion Matrix
- Classification Report

### 7. Save Model
Saved inside `models/svm_model.pkl`

---

## 📊 Example of Evaluation Output

```
ACCURACY: 1.0000

CONFUSION MATRIX:
[[796   0]
 [  0 772]]

CLASSIFICATION REPORT:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       796
           1       1.00      1.00      1.00       772
```

SVM achieves **100% accuracy** on this dataset.

---

## 🧠 Why SVM for This Project?

- Works extremely well for high-dimensional categorical data
- RBF kernel handles non-linear relationships
- Strong classifier with excellent separation capability

This specific mushroom dataset is known to be perfectly separable, making SVM an ideal choice.

---

## 🔍 Files Explained

| File | Purpose |
|------|---------|
| `main.py` | Runs the entire ML pipeline |
| `eda.py` | Loads dataset + prints info |
| `preprocess.py` | Encodes data + splits into train/test |
| `visualize.py` | Generates and saves all graphs |
| `train_svm.py` | Trains SVM, evaluates, and saves model |

---

## ✔ Requirements

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Or use a `requirements.txt` file:

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Then install:

```bash
pip install -r requirements.txt
```

---

## ✔ Final Notes

This project is:
- Clean and modular
- Perfect for assignments
- Ready for GitHub
- Fully automated
- Easy to understand and extend

---

## 📝 License

This project is open-source and available for educational purposes.

---

**Happy Coding! 🍄✨**