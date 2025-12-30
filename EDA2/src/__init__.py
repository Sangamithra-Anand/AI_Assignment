"""
This file marks the 'src' folder as a Python package.

It allows us to import modules like:
    from src.load_data import load_dataset
    from src.preprocess import preprocess_data
    from src.feature_engineering import engineer_features
    from src.feature_selection import feature_selection_pipeline
    from src.visualization import visualization_pipeline

Having __init__.py makes the project modular and easier to maintain.
"""

# Optional exports — makes importing easier for main.py
from .load_data import load_dataset
from .preprocess import preprocess_data
from .feature_engineering import engineer_features
from .feature_selection import feature_selection_pipeline
from .visualization import visualization_pipeline
from .utils import ensure_directory, save_text_report, print_section


