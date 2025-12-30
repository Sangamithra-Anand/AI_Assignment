# ======================================================================
# PROBLEM STATEMENT (Short Version)
# ======================================================================
# A company wants to find the average durability of its print-heads.
# Since testing a print-head destroys it, only 15 print-heads were tested.
#
# Using these 15 durability values, we must:
#   1. Build a 99% confidence interval using the SAMPLE SD (t-method).
#   2. Build another 99% confidence interval using POPULATION SD (z-method),
#      assuming the true standard deviation σ = 0.2.
#
# Goal: Estimate a safe range where the true average durability lies.
# ======================================================================


# -------------------------------------------------------------
# Step 1: Bring Python's tools
# -------------------------------------------------------------
import numpy as np              # Calculator for numbers
from scipy import stats         # Tools for t-values and z-values
import matplotlib.pyplot as plt # For drawing graphs


# -------------------------------------------------------------
# Step 2: Load our 15 durability test results
# -------------------------------------------------------------
data = np.array([
    1.13, 1.55, 1.43, 0.92, 1.25,
    1.36, 1.32, 0.85, 1.07, 1.48,
    1.20, 1.33, 1.18, 1.22, 1.29
])


# -------------------------------------------------------------
# Step 3: Let Python find basic information about the data
# -------------------------------------------------------------
n = len(data)        # How many print-heads were tested (15)
mean = np.mean(data) # Average durability
s = np.std(data, ddof=1)  
# Spread of the data (sample SD). ddof=1 is correct for samples


# -------------------------------------------------------------
# Step 4: First range — 99% CI using SAMPLE SD (t-method)
# -------------------------------------------------------------
alpha = 0.01        # For 99% confidence
df = n - 1          # Degrees of freedom

t_crit = stats.t.ppf(1 - alpha/2, df)     # Special t-value
margin_t = t_crit * (s / np.sqrt(n))      # How much above/below the mean
ci_t = (mean - margin_t, mean + margin_t) # The safe range (t-method)


# -------------------------------------------------------------
# Step 5: Second range — 99% CI using POPULATION SD (z-method)
# -------------------------------------------------------------
sigma = 0.2   # True standard deviation (given in problem)

z_crit = stats.norm.ppf(1 - alpha/2)        # Special z-value
margin_z = z_crit * (sigma / np.sqrt(n))    # Margin for z-method
ci_z = (mean - margin_z, mean + margin_z)   # Safe range (z-method)


# -------------------------------------------------------------
# Step 6: Print the results
# -------------------------------------------------------------
print("===============================================")
print("99% CI using SAMPLE SD (t-method):")
print(f"Lower Bound: {ci_t[0]:.4f}")
print(f"Upper Bound: {ci_t[1]:.4f}")
print("===============================================\n")

print("===============================================")
print("99% CI using POPULATION SD (z-method):")
print(f"Lower Bound: {ci_z[0]:.4f}")
print(f"Upper Bound: {ci_z[1]:.4f}")
print("===============================================\n")


# -------------------------------------------------------------
# Step 7: Draw a picture to help visualize the data and CI
# -------------------------------------------------------------
plt.figure(figsize=(10, 5))

plt.scatter(range(n), data, color='blue', label='Data Points') # Plot each value
plt.axhline(mean, color='black', linestyle='--', label='Average Durability')
plt.axhline(ci_t[0], color='red', linestyle=':', label='99% CI Lower (t)')
plt.axhline(ci_t[1], color='red', linestyle=':', label='99% CI Upper (t)')

plt.title("Durability Data with 99% Confidence Interval (t-method)")
plt.xlabel("Sample Number")
plt.ylabel("Durability (Millions of Characters)")
plt.legend()
plt.show()

