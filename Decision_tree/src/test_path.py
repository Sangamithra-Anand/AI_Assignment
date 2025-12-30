import os
import pandas as pd

path = "data/raw/heart_disease.xlsx"
print("FULL PATH:", os.path.abspath(path))

df = pd.read_excel(path)
print("SHAPE:", df.shape)
print("COLUMNS:", df.columns)
