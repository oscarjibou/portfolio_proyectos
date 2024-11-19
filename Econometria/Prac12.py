"""
1. Clean the data
2. Define the linear model
3. Check multicoliniarity with VIF and correlation matrix (cuantitatve variables)
4. Fit the model
5. Check the parameters, p-values, etc.
6. Check the residuals (white noise)
7. Interpret the results
8. Forecast
* For comparing models, use AIC, BIC, and R^2-adjusted

---
For time series:
- Exponential smoothing (no tendency or seasonality)
- Holt (tendency)
- Holts-Winters (tendency and seasonality)
SARIMA (tendency and seasonality) - Stockastic process (Media 0, Varianza constante, Covarianza constante)
1. Differencing for removing tendency and seasonality: d and D
2. Check autocorrelation and partial autocorrelation: p, q and P, Q, S
3. Choose the best model (AIC, BIC, std. error)
4. Check the residuals (white noise)
6. Forecast and check if the model is good (Average error, not infraestimation or sobreestimation)
"""
import os
import pandas as pd
import statsmodels.api as sm

from utilities import (
    autocorrelation_plots,
    backward_elimination,
    check_white_noise,
    residues_plot,
    time_plot,
)
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX


############################# SET UP #############################
threshold = 0.05  # Significance level

save_path = os.path.join(os.path.expanduser("~"), "Downloads")  # Path to save the plots

file_path = "Econometria/MRD021a.csv"  # Load the CSV file

data = pd.read_csv(file_path).dropna()  # Drop all NaN values from table
print(data.head(10))

variable_name = "CONSUMOG"  # Defining the dependent variable name

############################# VISUALIZATION #############################

time_plot(
    data["obs"],
    data[variable_name],
    variable_name=variable_name,
    ylim=data[variable_name].min(),
    save_path=save_path,
)

############################# LINEAR MODEL #############################

data["PARQUE^2"] = data["PARQUE"] ** 2

X = data[
    [
        "PARQUE",
        "PARQUE^2",
    ]
]
X = sm.add_constant(X)
print(X.head())
linear_model = sm.OLS(data[variable_name], X).fit()

print(linear_model.summary())

data["RESIDUES"] = linear_model.resid
print(linear_model.resid)
data["FITTED VALUES"] = linear_model.fittedvalues

print(data.head())

for column in data.columns:
    if column not in ["RESIDUES", "obs"]:
        residues_plot(
            residues=data["RESIDUES"],
            variable=data[column],
            variable_name=column,
            save_path=save_path,
        )
    else:
        continue

autocorrelation_plots(
    series=data["RESIDUES"],
    lags=6,
    variable_name=data["RESIDUES"].name,
    save_path=save_path,
)
############################# NEW MODEL #############################

data["DUMMY 1992"] = data["obs"].apply(lambda x: 1 if x == 1992 else 0)
data["DUMMY*PARQUE"] = data["DUMMY 1992"] * data["PARQUE"]

X = data[
    [
        "PARQUE",
        "DUMMY 1992",
        "DUMMY*PARQUE",
    ]
]
linear_model = sm.OLS(data[variable_name], sm.add_constant(X)).fit()

data["New Model Residues"] = linear_model.resid

check_white_noise(data["New Model Residues"], exog=data["obs"])


residues_plot(
    residues=data["New Model Residues"],
    variable=data["obs"],
    variable_name="obs",
    save_path=save_path,
)

autocorrelation_plots(
    series=data["New Model Residues"],
    lags=6,
    variable_name=data["New Model Residues"].name,
    save_path=save_path,
)


auto_arima(
    data["New Model Residues"], seasonal=False, m=1, trace=True, error_action="ignore"
)

model = SARIMAX(
    data["New Model Residues"],
    order=(1, 1, 0),
    exog=data[["PARQUE", "DUMMY 1992", "DUMMY*PARQUE"]],
)


print(model.fit().summary())
