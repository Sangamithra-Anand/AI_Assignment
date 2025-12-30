"""
__init__.py
------------
This file turns the src/ folder into a Python package.

It also provides quick access to the main project modules.

For example:
from src import load_data, preprocess, train_model
"""

from .load_data import load_raw_dataset
from .preprocess import preprocess_data
from .eda import run_eda
from .feature_engineering import feature_engineering
from .train_model import train_decision_tree
from .evaluate import evaluate_model
from .visualize_tree import visualize_tree
from .utils import create_project_folders
