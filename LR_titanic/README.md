# 🚢 Titanic Survival Prediction (Logistic Regression)

A complete **end-to-end Machine Learning pipeline** built using Python, Scikit-Learn, and Streamlit.

This project predicts whether a passenger on the Titanic survived or not using a **Logistic Regression** model.  
It includes:

✅ Data Preprocessing  
✅ Model Training  
✅ Model Evaluation (metrics + plots)  
✅ Streamlit Web App for live predictions  
✅ Automated Command-Line Runner (`main.py`)  

---

## 📂 Project Structure

```
titanic_logistic_regression/
│
├── data/                       # Raw dataset (Titanic_train.csv, Titanic_test.csv)
│
├── src/                        # All backend ML code
│   ├── preprocessing.py        # Cleans and prepares dataset
│   ├── train_model.py          # Trains Logistic Regression model
│   ├── evaluate.py             # Evaluates model + saves reports & plots
│   ├── utils.py                # Helper functions
│   └── main.py                 # Command-line pipeline runner
│
├── models/
│   └── logistic_model.pkl      # Saved trained model (auto-generated)
│
├── output/
│   ├── plots/                  # Evaluation plots (auto-generated)
│   ├── reports/                # Classification reports (auto-generated)
│   └──clean_train.csv         # Cleaned dataset (auto-generated)
│
├── streamlit_app            
│   ├── app.py                  # Streamlit UI for predictions
│
└── README.md
└── requirements.txt
```

---

## ⚙️ Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

This project includes an **automated command runner**:

```bash
python src/main.py
```

You will see:

```
1. Run Preprocessing
2. Train Model
3. Evaluate Model
4. Run ALL steps (Preprocess → Train → Evaluate → Launch Streamlit App)
5. Exit
```

---

## ⭐ Recommended Option (for full automation)

### Choose **OPTION 4**:

```
Run ALL steps (Preprocess → Train → Evaluate → Launch Streamlit App)
```

✔ Cleans dataset  
✔ Trains the model  
✔ Generates evaluation report  
✔ Launches Streamlit app automatically  
✔ Shows live model predictions  

> **NOTE:**  
> Streamlit runs in background.  
> To stop it:
> - Close the browser **AND**
> - Press **CTRL + C** in the main terminal

---

## 🌐 Running Streamlit Manually (Optional)

```bash
streamlit run streamlit_app.py
```

This will open the prediction UI in your browser.

---

## 📊 Streamlit App Features

### ✔ Single Passenger Prediction  
Enter values manually → app shows survival probability.

### ✔ Batch CSV Prediction  
Upload any CSV → app preprocesses, predicts, and allows download.

### ✔ Model Performance Metrics  
Reads evaluation report from `/output/reports`.

### ✔ Feature Importance  
Shows logistic regression coefficients visually.

---

## 🧠 Machine Learning Details

- **Model:** Logistic Regression (Scikit-Learn)  
- **Target:** Survived (0 = No, 1 = Yes)  
- **Preprocessing includes:**  
  - Filling missing Age/Fare with median  
  - Encoding Sex, Embarked  
  - Adding FamilySize feature  
  - Dropping noisy columns  
  - Final NaN cleanup  

- **Evaluation Metrics:**  
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  
  - Confusion Matrix  
  - Probability histograms  

All results are saved to **/output/** automatically.

---

**Happy Learning! 🚀**