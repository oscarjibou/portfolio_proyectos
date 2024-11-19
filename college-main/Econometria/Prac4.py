import xml.etree.ElementTree as ET
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from utilities import (
    compute_residuals,
    eliminate_max_correlated_with_target,
)


# ----------------------------------------- DATA PREPARATION -----------------------------------------#
# Parse the XML data
tree = ET.parse("/Users/carlosedm10/projects/college/Econometria/MRL028tc.gdt")
root = tree.getroot()

# Extracting variable names from the XML
variable_names = [variable.get("name") for variable in root.findall(".//variable")]

# Extracting all the data from the "obs" tags
all_obs_data = [obs.text.split() for obs in root.findall(".//observations/obs")]

# Convert the extracted data to a DataFrame using the variable names as columns
df = pd.DataFrame(all_obs_data, columns=variable_names)

# Convert to appropriate data types
df = df.apply(pd.to_numeric)

df.head()

# Exclude the TRIM variables and compute the correlation matrix
correlation_matrix = df.drop(
    columns=["TRIM", "TRIM1", "TRIM2", "TRIM3", "TRIM4"]
).corr()

print(correlation_matrix)

# Apply the function
df_reduced, removed_variable = eliminate_max_correlated_with_target(df, "MUERTOS")

print(f"Removed variable: {removed_variable}")


# Plot the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, square=True
)
plt.title("Correlation Matrix")
plt.show()

# ----------------------------------------- REGRESSION -----------------------------------------#

# Prepare data for regression with 'MUERTOS' as dependent variable
y = df_reduced["MUERTOS"]
X = df_reduced.drop(["MUERTOS", "TRIM", "TRIM1", "TRIM2", "TRIM3", "TRIM4"], axis=1)
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()  # ordinary least squares model

# Get the summary of the regression
summary = model.summary()

print(summary)
# ----------------------------------------- RESIDUALS -----------------------------------------#
# Calculating residuals
residuals = model.resid
fitted = model.fittedvalues

residuals_muertos = compute_residuals("MUERTOS", ["PARQUE"], df)
residuals_accidentes = compute_residuals("ACCIDENTES", ["PARQUE"], df)

# Plotting residuals vs fitted values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=fitted, y=residuals)
plt.axhline(y=0, color="red", linestyle="--")
plt.title("Residuals vs Fitted Values")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.show()


# Histogram of residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=20)
plt.title("Histogram of Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.show()

# Q-Q plot for residuals
plt.figure(figsize=(10, 6))
sm.qqplot(residuals, line="45", fit=True)
plt.title("Q-Q Plot of Residuals")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(
    df_reduced.index, residuals, marker="o", linestyle="-", color="blue", alpha=0.5
)  # Line plot with points
plt.axhline(y=0, color="r", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Residuals")
plt.title("Residuals over Time")
plt.show()

# Plot residuals for MUERTOS against PARQUE
fig, ax1 = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

# Scatter plot for MUERTOS residuals against PARQUE
ax1.scatter(df["PARQUE"], residuals_muertos, color="blue", alpha=0.7)
ax1.axhline(y=0, color="red", linestyle="--")
ax1.set_title("Residuals for MUERTOS vs PARQUE")
ax1.set_ylabel("Residuals")
ax1.grid(True)
plt.show()
