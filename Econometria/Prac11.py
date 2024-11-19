import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import ARIMA, auto_arima
from matplotlib import pyplot as plt
from utilities import (
    check_white_noise,
    format_diagnostics,
    time_plot,
)

############################################# DEFINITIONS #############################################

# Significance level
threshold = 0.05

# Load the CSV file
file_path = "Econometria/TASA_PARO.csv"

data = pd.read_csv(file_path).dropna()  # Drop all NaN values from table
# Split the "obs" column into year and quarter
data[["Year", "Quarter"]] = data["obs"].str.split("Q", expand=True)

# Create a new column "Date" with the first day of the respective quarter

print(data.head())

variable_name = "PARO"  # Defining the variable name

print(f"df_1985: {data}")
y = data[variable_name]
x = data["obs"]


######################################### Pregunta 2 #########################################

# Time series plot
time_plot(x, y, variable_name=variable_name, ylim=y.min())

######################################### Pregunta 3 #########################################

lags = 12

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(y, ax=ax1, lags=lags, title=f"Autocorrelation for time series")
plot_pacf(y, ax=ax2, lags=lags, title=f"Partial Autocorrelation for time series")

plt.tight_layout()
plt.show()

# ----------------------------- Diferencing -----------------------------#

dy = y.diff().dropna()  # First difference
dy = dy.diff(periods=4).dropna()  # Seasonal difference for quarterly data

time_plot(dy, variable_name=variable_name)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(dy, ax=ax1, lags=lags, title="Autocorrelation for differences")
plot_pacf(dy, ax=ax2, lags=lags, title="Partial Autocorrelation for differences")

plt.tight_layout()
plt.show()

######################################### Pregunta 4 #########################################

p = 1
d = 1
q = 0
P = 1
D = 1
Q = 0
S = 4

model = SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, S))
results = model.fit()
print(results.summary())

# Access the residuals using the 'resid' attribute
residuals = results.resid

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(residuals, ax=ax1, lags=lags, title="Autocorrelation for residues 1")
plot_pacf(residuals, ax=ax2, lags=lags, title="Partial Autocorrelation for residues 1")

plt.tight_layout()
plt.show()

######################################### Preguntas 5-8 #########################################
p = 1
d = 1
q = 0
P = 1
D = 1
Q = 0
S = 4

model = SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, S))
results = model.fit()

print(format_diagnostics(check_white_noise(residuals, y, alpha=0.05)))
######################################### RESIDUALS #########################################
print("\n----------------------White Noise ----------------------\n")
# Access the residuals using the 'resid' attribute
residuals = results.resid

time_plot(residuals, variable_name="Residuals")

print(format_diagnostics(check_white_noise(residuals, y, alpha=0.05)))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(residuals, ax=ax1, lags=lags, title="Autocorrelation for residues 2")
plot_pacf(residuals, ax=ax2, lags=lags, title="Partial Autocorrelation for residues 2")

plt.tight_layout()
plt.show()
X = np.arange(1, len(residuals) + 1)

# Estimar el modelo con errores estándar robustos
residual_linear_model = sm.OLS(residuals**2, X).fit()
print(residual_linear_model.summary())


f_pvalue = residual_linear_model.f_pvalue  # P-value for the F-statistic

if f_pvalue < threshold:
    print(
        f"P-value for the F-statistic is {f_pvalue} and it is less than {threshold}. The model has heteroscedasticity. Reject H0."
    )
else:
    print(
        f"P-value for F-statistic is {f_pvalue} and it is greater than {threshold}. The model does not have heteroscedasticity. Accept H0."
    )
