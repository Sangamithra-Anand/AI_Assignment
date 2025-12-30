import pandas as pd

# Load your dataset
df = pd.read_csv("sales_data_with_discounts.csv")

# Select only numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64'])

# Function to apply Z-score standardization
def zscore_standardize(column):
    mean = column.mean()
    std = column.std()
    return (column - mean) / std

# Create a new DataFrame for standardized values
df_zscore = num_cols.apply(zscore_standardize)

# Save to new CSV
df_zscore.to_csv("sales_zscore_standardized.csv", index=False)

print("Z-score Standardization completed!")
print("File saved as sales_zscore_standardized.csv")
print("\nPreview:")
print(df_zscore.head())
