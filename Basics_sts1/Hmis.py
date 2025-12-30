import pandas as pd

df = pd.read_csv("sales_data_with_discounts.csv")

# Numerical columns → fill with mean
num_cols = df.select_dtypes(include=['int64', 'float64'])
df[num_cols.columns] = df[num_cols.columns].fillna(df[num_cols.columns].mean())

# Categorical columns → fill with mode
cat_cols = df.select_dtypes(include=['object'])
df[cat_cols.columns] = df[cat_cols.columns].fillna(df[cat_cols.columns].mode().iloc[0])

df.to_csv("sales_cleaned.csv", index=False)

print("Missing values handled and saved as sales_cleaned.csv")
