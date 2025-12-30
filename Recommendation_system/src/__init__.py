"""
src package
-----------
This file marks the 'src' directory as a Python package.

PURPOSE:
- Allows importing modules using: 
      from src.module_name import something
- Helps maintain a clean, modular project structure.

We are NOT importing all modules here intentionally.
Each module should be imported only when needed 
to avoid circular import issues.
"""

__all__ = [
    "load_data",
    "preprocess",
    "feature_engineering",
    "similarity",
    "recommend",
    "evaluation"
]
