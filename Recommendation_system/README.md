# 🎌 Anime Recommendation System

**Content-Based Recommender (Python 3.13 – Custom TF-IDF + Cosine Similarity)**

This project is a complete Anime Recommendation System built entirely using Python, without heavy ML libraries. It uses a custom TF-IDF-like feature extractor, manual numeric scaling, and cosine similarity to generate high-quality recommendations.

The system includes a full pipeline:
- ✔ Load → Preprocess → Feature Engineering → Similarity Computation → Recommendation → Evaluation
- ✔ RAM-optimized & works smoothly even on low-spec PCs
- ✔ Fully menu-driven CLI interface
- ✔ Clean logs + timers for performance tracking
- ✔ Compatible with Python 3.13

---

## 📁 Project Structure

```
Recommendation-system/
│
├── data/
│   ├── raw/
│   │   └── anime.csv
│   └── processed/
│       ├── cleaned_anime.csv
│       └── features_matrix.pkl
│
├── models/
│   └── feature_config.pkl
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── load_data.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── similarity.py
│   ├── recommend.py
│   ├── evaluate.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
│
└── README.md
```

---

## 🚀 Features

### ✔ 1. Data Preprocessing
- Removes duplicates
- Fixes non-numeric values
- Converts invalid values (`"?"`, `"unknown"`, `"N/A"`)
- Fills missing numeric fields using median
- Saves clean dataset → `data/processed/cleaned_anime.csv`

### ✔ 2. Custom Feature Engineering (TF-IDF + Numeric Scaling)
- Builds vocabulary from anime genres
- Creates TF-IDF-like vectors without scikit-learn
- Normalizes numeric features:
  - `rating`
  - `members`
  - `episodes`
- Concatenates genre + numeric vectors
- Saves artifacts →
  - `models/feature_config.pkl`
  - `data/processed/features_matrix.pkl`

⚡ **Fast & RAM-safe** (12232 anime processed in ~0.05s)

### ✔ 3. Cosine Similarity (Memory Efficient)
- Uses dot product formula
- No massive similarity matrix stored in RAM
- Computes similarity only when needed

### ✔ 4. Anime Recommendation
Provides **TOP-N similar anime** based on:
- Genre similarity
- Rating similarity
- Member count
- Episode count

If anime not found → suggests close matches.

### ✔ 5. System Evaluation
Uses simple precision, recall, and F1-score to validate consistency.

### ✔ 6. Fully Menu-Driven CLI
Example:

```
============================================================
          ANIME RECOMMENDATION SYSTEM — MENU
============================================================
1. Load Raw Dataset
2. Preprocess Dataset
3. Feature Engineering
4. (Removed)
5. Get Recommendations
6. Evaluate System
7. Run FULL PIPELINE
8. Exit
============================================================
```

---

## 🛠 Installation

### 1. Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux
```

### 2. Install Required Libraries

Create a `requirements.txt`:

```
pandas
numpy
```

Install:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Project

### Run Main Program

```bash
python -m src.main
```

---

## 🎯 Usage Examples

### ⭐ Get Recommendations

```
Enter your choice: 5
Enter anime title: Naruto
```

**Output:**

```
Recommended Anime | Genre             | Rating | Similarity Score
------------------------------------------------------------------
Bleach            | Action, Shounen   | 7.8    | 0.82
One Piece         | Action, Adventure | 8.6    | 0.80
...
```

### 📊 Evaluation Output (Example)

```
Precision: 1.0
Recall: 1.0
F1-Score: 1.0
```

### 🔧 Full Pipeline Example

```
Enter your choice: 7
[INFO] Running FULL PIPELINE...
- Preprocessing complete
- Feature Engineering complete
- Artifacts saved
Pipeline completed ✔ in 0.11 seconds
```

---

## 📌 Notes

- Works perfectly on **Python 3.10 → 3.13**
- **No heavy ML libraries** → Ultra-fast & lightweight
- Dataset must contain:
  - `name`
  - `genre`
  - `rating`
  - `members`
  - `episodes`

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or feedback, reach out via GitHub issues or email.

---

**Happy Recommending! 🎬✨**