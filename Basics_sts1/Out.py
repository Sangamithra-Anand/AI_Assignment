import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("sales_data_with_discounts.csv")

# Select only numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64'])

# Loop through each numerical column
for col in num_cols:

    # --- IQR Outlier Calculation ---
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    # Print outlier details
    print(f"\nColumn: {col}")
    print(f"Lower Bound: {lower}, Upper Bound: {upper}")
    print(f"Outliers Count: {len(outliers)}")

    # --- Boxplot Visualization ---
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Outlier Visualization for {col}")
    plt.show()
