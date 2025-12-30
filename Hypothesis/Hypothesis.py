"""
Hypothesis Testing Assignment (Updated with Automatic Critical Value)
---------------------------------------------------------------------

This program checks whether the weekly operating cost is HIGHER than the
model W = 1000 + 5X using a one-tailed Z-test.

We now compute the CRITICAL VALUE automatically using SciPy:
    norm.ppf(1 - alpha)
"""

import math
from scipy.stats import norm   

# -----------------------------------------------------------
# STEP 1: Hypotheses
# -----------------------------------------------------------
print("STEP 1: Hypotheses")
print("H0 (Null Hypothesis): Weekly operating cost is NOT higher than the model predicts.")
print("H1 (Alternative Hypothesis): Weekly operating cost IS higher than the model predicts.\n")

# -----------------------------------------------------------
# STEP 2: Given Data
# -----------------------------------------------------------
sample_mean = 3050
units_mean = 600
mu = 1000 + 5 * units_mean     # Theoretical mean = 4000
sigma = 5 * 25                 # Standard deviation = 125
n = 25

print("STEP 2: Data Used")
print(f"Sample Mean = {sample_mean}")
print(f"Theoretical Mean (μ) = {mu}")
print(f"Standard Deviation (σ) = {sigma}")
print(f"Sample Size (n) = {n}\n")

# -----------------------------------------------------------
# STEP 3: Test Statistic
# -----------------------------------------------------------
standard_error = sigma / math.sqrt(n)
test_stat = (sample_mean - mu) / standard_error

print("STEP 3: Test Statistic")
print(f"Standard Error = {standard_error}")
print(f"Test Statistic (Z) = {test_stat}\n")

# -----------------------------------------------------------
# STEP 4: AUTOMATIC Critical Value (SciPy)
# -----------------------------------------------------------
alpha = 0.05   # 5% significance level for right-tailed test

# Method 1: AUTOMATIC calculation using SciPy
critical_value = norm.ppf(1 - alpha)    # Gives 1.64485... (≈ 1.645)

print("STEP 4: Critical Value (Automatic Using SciPy)")
print(f"Alpha Level (α) = {alpha}")
print(f"Critical Z Value = {critical_value}\n")

# -----------------------------------------------------------
# STEP 5: Decision
# -----------------------------------------------------------
print("STEP 5: Decision")

if test_stat > critical_value:
    print("Test Statistic is GREATER than Critical Value.")
    print("Decision: Reject H0")
    print("Conclusion: Weekly operating cost IS higher than the model predicts.")
else:
    print("Test Statistic is NOT greater than Critical Value.")
    print("Decision: Do NOT reject H0")
    print("Conclusion: There is NO evidence that weekly operating cost is higher.\n")

