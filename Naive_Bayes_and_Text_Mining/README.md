# 📘 Blog Text Classification & Sentiment Analysis  
### 🔍 Naive Bayes + TF-IDF | End-to-End NLP Pipeline

This project performs **Text Classification** and **Sentiment Analysis** on a dataset of 2000 blog posts.  
A complete machine-learning pipeline was built using **TF-IDF**, **Naive Bayes**, and **TextBlob** for sentiment scoring.

---

## 🚀 Project Features

### ✔ Text Classification (Supervised ML)
- Cleans raw text
- Converts text into numerical vectors using **TF-IDF**
- Trains a **Multinomial Naive Bayes Classifier**
- Achieves **85%+ accuracy**
- Generates evaluation metrics and confusion matrix

### ✔ Sentiment Analysis (Unsupervised NLP)
- Computes polarity score using **TextBlob**
- Categorizes sentiment into:
  - **Positive**
  - **Negative**
  - **Neutral**
- Saves full sentiment output + summary report

### ✔ Fully Modular Code
- Clean architecture inside the `/src/` folder
- Every step runs independently
- Also supports **Full Pipeline Execution** with a single menu option

---

## 📁 Project Structure

```
project_root/
│
├── data/
│   ├── raw/
│   │   └── blogs_categories.csv
│   └── processed/
│       └── cleaned_blogs.csv
│
├── models/
│   ├── naive_bayes_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── reports/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── sentiment_summary.txt
│
├── outputs/
│   └── sentiment_results.csv
│
├── src/
│   ├── __init__.py
│   ├── load_data.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate.py
│   ├── sentiment_analysis.py
│   └── main.py
│
├── requirements.txt
└── README.md
```

---

## 🔧 Installation & Setup

### 1️⃣ Install Required Packages
```bash
pip install -r requirements.txt
```

### 2️⃣ Download NLTK Resources
Run Python shell:
```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
```

### 3️⃣ Download TextBlob Corpora
```bash
python -m textblob.download_corpora
```

---

## ▶️ Running the Project

Start the pipeline:
```bash
python src/main.py
```

You will see a menu:
```
1. Load Raw Dataset
2. Preprocess Dataset
3. Train Naive Bayes Model
4. Evaluate Model
5. Perform Sentiment Analysis
6. RUN FULL PIPELINE (Recommended)
7. Exit
```

### ⭐ Recommended: Choose Option 6
Runs the complete pipeline:
- ✔ Preprocessing
- ✔ TF-IDF Vectorization
- ✔ Naive Bayes Training
- ✔ Model Evaluation
- ✔ Sentiment Analysis

---

## 📊 Output Files Generated

### 📄 Cleaned Data
```
data/processed/cleaned_blogs.csv
```

### 🤖 Model Files
```
models/naive_bayes_model.pkl
models/tfidf_vectorizer.pkl
```

### 📈 Evaluation Reports
```
reports/classification_report.txt
reports/confusion_matrix.png
```

### 😊 Sentiment Analysis
```
outputs/sentiment_results.csv
reports/sentiment_summary.txt
```

---

## 📌 Model Performance Summary

Using Multinomial Naive Bayes:

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~0.855 |
| Precision | ~0.862 |
| Recall    | ~0.855 |
| F1-Score  | ~0.855 |

The model performs well for text classification tasks with TF-IDF features.

---

## 🛠 Technologies Used

- Python 3
- Pandas
- Scikit-Learn
- NLTK
- TextBlob
- Matplotlib

---

## 📜 License

This project is for academic and learning purposes.