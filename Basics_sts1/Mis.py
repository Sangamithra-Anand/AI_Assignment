import pandas as pd

df = pd.read_csv("sales_data_with_discounts.csv")

print("\nMissing Values Count:")
print(df.isnull().sum())

print("\nMissing Values Percentage:")
print(df.isnull().mean() * 100)
