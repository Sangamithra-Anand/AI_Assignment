"""
__init__.py
-----------
This file allows simplified importing of modules from the src package.
"""

from .load_data import load_glass_data
from .eda import run_eda
from .visualize import run_visualizations
from .preprocess import preprocess_glass_data
from .train_random_forest import train_random_forest
from .bagging_boosting import run_bagging_and_boosting
from .evaluate import evaluate_model
from .utils import ensure_folder_exists, log_message, start_timer, end_timer

__all__ = [
    "load_glass_data",
    "run_eda",
    "run_visualizations",
    "preprocess_glass_data",
    "train_random_forest",
    "run_bagging_and_boosting",
    "evaluate_model",
    "ensure_folder_exists",
    "log_message",
    "start_timer",
    "end_timer"
]
