"""
src package for KNN_Zoo_Classification.

Purpose:
- Marks the `src` folder as a Python package.
- Provides a small, safe convenience import area if you want to
  import common functions like `load_zoo_data` via `from src import load_zoo_data`.

Notes:
- Keep this file minimal to avoid side-effects at import time.
- If you prefer NOT to expose these at package level, you can leave this file almost empty
  (only keep the docstring).
"""

# Package version (optional)
__version__ = "0.1.0"

# Expose commonly-used functions for convenience.
# Importing inside try/except avoids hard errors if a module is missing during some tests.
try:
    from .load_data import load_zoo_data
    from .preprocess import preprocess_data
    from .knn_model import train_knn, find_best_k
    from .evaluate import evaluate_model
    from .visualize import generate_all_visualizations
except Exception:
    # If import fails (e.g., running tests before all files exist), don't crash on package import.
    pass

# __all__ controls what `from src import *` will export.
__all__ = [
    "load_zoo_data",
    "preprocess_data",
    "train_knn",
    "find_best_k",
    "evaluate_model",
    "generate_all_visualizations",
]

