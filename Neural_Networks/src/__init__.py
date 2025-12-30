"""
__init__.py
-----------
This file marks the 'src' directory as a Python package.

Why we need this file:
- Python only treats a folder as a package if __init__.py exists.
- It allows you to import modules like:
      from src.preprocess import preprocess_data
      from src.train import train_baseline_model
- Without this file, relative imports inside the project may fail.

We can also optionally expose commonly used functions here,
so they can be imported directly from src.
"""

# OPTIONAL: Import commonly used functions for easier access
# (You can remove these if you prefer a cleaner namespace)

from .data_loader import load_raw_data
from .preprocess import preprocess_data
from .model_builder import build_ann_model
from .train import train_baseline_model
from .tune_hyperparameters import run_hyperparameter_search
from .evaluate import evaluate_baseline_and_tuned_models

# Now you can do:
#   from src import train_baseline_model
# instead of:
#   from src.train import train_baseline_model

