from pathlib import Path

# The root folder of the project (cardio-eda)
ROOT = Path(__file__).resolve().parents[1]

# Folders inside the project
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
SUMMARY_DIR = OUTPUT_DIR / "summary"

# Full path to your CSV dataset
DATA_FILE = DATA_DIR / "Cardiotocographic.csv"
