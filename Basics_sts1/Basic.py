import pandas as pd

# Load dataset
df = pd.read_csv("sales_data_with_discounts.csv")

# Show first rows
print("First 5 rows:")
print(df.head())

# Show info
print("\nDataset Info:")
print(df.info())

# Show summary statistics
print("\nSummary Statistics:")
print(df.describe())
