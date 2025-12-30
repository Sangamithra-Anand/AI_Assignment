# 🧠 Neural Network Classification Project

**Alphabet Recognition using Artificial Neural Networks (ANN)**

This project builds a complete Machine Learning pipeline for classifying handwritten alphabets using a fully connected Artificial Neural Network. It includes data preprocessing, model training, hyperparameter tuning, evaluation, and visualization, all controlled through a simple **MENU-DRIVEN SYSTEM**.

---

## 🚀 Features

### ✅ Full ML Pipeline (End-to-End)

Includes all major machine learning stages:

1. **Data Loading**
2. **Preprocessing** (cleaning, scaling, encoding)
3. **Building ANN Models**
4. **Training the baseline model**
5. **Hyperparameter Tuning**
6. **Evaluation & Metrics**
7. **Visualizations** (loss curves, accuracy curves, confusion matrix)

---

## 🏗️ Project Folder Structure

```
Neural_Networks/
│
├── data/
│   ├── raw/               # Raw dataset (input manually)
│   ├── interim/           # Cleaned dataset (auto)
│   └── processed/         # Final processed dataset (auto)
│
├── models/                # Saved ANN models (auto)
├── reports/               # Logs, tuning summary, training report (auto)
├── output/
│   ├── metrics/           # JSON metric files (auto)
│   └── figures/           # Plots (auto)
│
├── src/
│   ├── main.py            # Main menu-driven controller
│   ├── config.py          # Configurations & paths
│   ├── data_loader.py     # Load dataset
│   ├── preprocess.py      # Clean & scale data
│   ├── model_builder.py   # Build ANN model
│   ├── train.py           # Train baseline model
│   ├── tune_hyperparameters.py  # Hyperparameter tuning
│   ├── evaluate.py        # Evaluate baseline + tuned models
│   ├── visualize_results.py # Plot figures
│   └── utils.py           # Helper functions
│
└── requirements.txt       # Python dependencies
```

---

## 🧪 Dataset

The dataset used is **Alphabets_data.csv**, containing features extracted from handwritten alphabet images using statistical properties (like height, width, edges, bars, etc.).

- **Input columns**: 16 numerical features
- **Target column**: `letter` (A–Z → 26 classes)
- **Total rows**: 20,000 examples

---

## ▶️ How to Run the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Main Application

```bash
python src/main.py
```

---

## 🧭 Main Menu Options

When you run `main.py`, you get a user-friendly menu:

```
==========================
      MAIN MENU
==========================
1. Preprocess Data
2. Train Baseline Model
3. Run Hyperparameter Tuning
4. Evaluate Models
5. Visualize Results
6. Run ALL Steps (Full Pipeline)
7. Exit
```

### 💡 Recommended Usage Order

**1 → 2 → 3 → 4 → 5** (or select option **6** to run everything automatically)

---

## ⚙️ Hyperparameter Tuning (Improved)

Your tuning includes:

- 🔹 **Progress Bar + ETA**
- 🔹 **Colored Output**
- 🔹 **Early Stopping** (val_accuracy ≥ 0.93)

### 🔹 Saves:

- Best model → `models/best_model.h5`
- All tuning results → `reports/hyperparameter_search_results.csv`
- Best metrics → `output/metrics/tuned_metrics.json`
- Summary → `reports/best_hyperparameters.txt`

---

## 📊 Visualizations

The following plots are automatically saved:

- **Loss Curve** → `loss_curve.png`
- **Accuracy Curve** → `accuracy_curve.png`
- **Confusion Matrix** → `confusion_matrix.png`

All inside:

```
output/figures/
```

---

## 📈 Baseline Performance (Typical)

| Metric          | Value                              |
| --------------- | ---------------------------------- |
| Test Accuracy   | ~92–93%                            |
| Test Loss       | ~0.21                              |
| Tuned Accuracy  | ≥95% (depending on parameters)     |

---

## 💼 Skills Demonstrated

This project shows proficiency in:

- ✅ Neural Network Architecture
- ✅ Data Preprocessing Pipelines
- ✅ Hyperparameter Optimization
- ✅ Model Evaluation
- ✅ Python & TensorFlow
- ✅ Modular ML Project Structure
- ✅ Automation / Menu-driven ML systems

**Great for portfolio, college project, and interview showcase.**

---

## 📝 License

This project is open-source and available for educational purposes.

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements!

## 📧 Contact

For questions or collaboration, feel free to reach out!

---

**Happy Learning! 🎓**