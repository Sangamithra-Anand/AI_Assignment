# -----------------------------------------
# 1. Import Required Libraries
# -----------------------------------------
import pandas as pd
import numpy as np

# -----------------------------------------
# 2. Load Your Dataset
# -----------------------------------------
# Replace the file name with your own dataset
df = pd.read_csv("sales_data_with_discounts.csv")  

# Show first 5 rows to understand the data
print("Dataset Preview:")
print(df.head(10))

# -----------------------------------------
# 3. Identify Numerical Columns
# -----------------------------------------
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("\nNumerical Columns:")
print(numerical_cols)

# -----------------------------------------
# 4. Calculate Basic Statistical Measures
# -----------------------------------------
mean_values = df[numerical_cols].mean()
median_values = df[numerical_cols].median()
mode_values = df[numerical_cols].mode().iloc[0]   # mode returns a dataframe
std_values = df[numerical_cols].std()
skew_values =df[numerical_cols].skew()
kurt_values =df[numerical_cols].kurt()


# -----------------------------------------
# 5. Display Results
# -----------------------------------------
print("\n===== Descriptive Statistics Results =====")

print("\nMean:")
print(mean_values)

print("\nMedian:")
print(median_values)

print("\nMode:")
print(mode_values)

print("\nStandard Deviation:")
print(std_values)

print("\nSkewness:")
print(skew_values)

print("\nKurtosis:")
print(kurt_values)

# -----------------------------------------
# 6. Additional Summary in One Line
# -----------------------------------------
print("\nPandas Built-in Describe() Summary:")
print(df[numerical_cols].describe())
