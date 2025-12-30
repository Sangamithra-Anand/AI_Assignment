"""
Chi-Square Test for Independence (Updated with Automatic Critical Value)
------------------------------------------------------------------------

This program tests whether Device Type (Thermostat vs Light)
is associated with Customer Satisfaction.

We now calculate the critical value AUTOMATICALLY using SciPy:
    chi2.ppf(1 - alpha, df)
"""

import numpy as np
from scipy.stats import chi2   

# -----------------------------------------------------------
# STEP 1: Hypotheses
# -----------------------------------------------------------
print("STEP 1: Hypotheses")
print("H0 (Null): Device Type and Satisfaction are INDEPENDENT.")
print("H1 (Alternative): Device Type and Satisfaction are ASSOCIATED.\n")

# -----------------------------------------------------------
# STEP 2: Observed Data (Given)
# -----------------------------------------------------------
observed = np.array([
    [50, 70],   # Very Satisfied
    [80, 100],  # Satisfied
    [60, 90],   # Neutral
    [30, 50],   # Unsatisfied
    [20, 50]    # Very Unsatisfied
])

print("STEP 2: Observed Frequency Table")
print(observed, "\n")

# -----------------------------------------------------------
# STEP 3: Expected Frequencies
# -----------------------------------------------------------
row_totals = observed.sum(axis=1)
col_totals = observed.sum(axis=0)
grand_total = observed.sum()

expected = np.outer(row_totals, col_totals) / grand_total

print("STEP 3: Expected Frequency Table")
print(expected, "\n")

# -----------------------------------------------------------
# STEP 4: Chi-Square Statistic
# -----------------------------------------------------------
chi_square = ((observed - expected) ** 2 / expected).sum()

print("STEP 4: Chi-Square Statistic")
print(f"Chi-Square Value = {chi_square}\n")

# -----------------------------------------------------------
# STEP 5: AUTOMATIC Critical Value (SciPy)
# -----------------------------------------------------------
alpha = 0.05
df = (observed.shape[0] - 1) * (observed.shape[1] - 1)

# Automatic critical value from SciPy
critical_value = chi2.ppf(1 - alpha, df)

print("STEP 5: Critical Value (Automatic Using SciPy)")
print(f"Degrees of Freedom = {df}")
print(f"Alpha Level (α) = {alpha}")
print(f"Critical Chi-Square Value = {critical_value}\n")

# -----------------------------------------------------------
# STEP 6: Decision
# -----------------------------------------------------------
print("STEP 6: Decision & Conclusion")

if chi_square > critical_value:
    print("Chi-Square is GREATER than Critical Value.")
    print("Decision: Reject H0")
    print("Conclusion: Device Type and Satisfaction ARE associated.")
else:
    print("Chi-Square is NOT greater than Critical Value.")
    print("Decision: Do NOT reject H0")
    print("Conclusion: There is NO evidence of association between Device Type and Satisfaction.\n")


