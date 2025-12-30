import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("sales_data_with_discounts.csv")

# Histogram for all numerical columns
df.hist(figsize=(12, 8))
plt.suptitle("Histograms")
plt.show()

# Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.select_dtypes(include=['int64', 'float64']))
plt.title("Boxplot - Outlier Detection")
plt.show()

# Pairplot
sns.pairplot(df.select_dtypes(include=['int64', 'float64']))
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
