import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from utilities import (
    autocorrelation_plots,
    check_white_noise,
    residues_plot,
    time_plot,
)
from statsmodels.tsa.statespace.sarimax import SARIMAX

############################# SET UP #############################
save_path = os.path.join(os.path.expanduser("~"), "Downloads")

# Load the CSV file
file_path = "Econometria/MRD060.csv"

data = pd.read_csv(file_path).dropna()  # Drop all NaN values from table
data["obs"] = pd.to_datetime(data["obs"], format="%YM%m")  # Format date
print(data.head(10))

variable_name = "VIAJEROSCVN"  # Defining the dependent variable name


############################# VISUALIZATION #############################

time_plot(
    data["obs"],
    data[variable_name],
    variable_name=variable_name,
    ylim=data[variable_name].min(),
    save_path=save_path,
)
############################# EJ1 #############################
data[f"log({variable_name})"] = np.log(data[variable_name])  # Logarithm of the variable
log = data[f"log({variable_name})"]
log_variable = data[f"log({variable_name})"]
log_variable = log_variable.diff().dropna()
log_variable = log_variable.diff(periods=12).dropna()

autocorrelation_plots(
    series=log_variable,
    lags=12,
    variable_name="log(VIAJEROSCVN)",
    save_path=save_path,
)

a_model_1 = SARIMAX(log, order=(0, 1, 1), seasonal_order=(0, 1, 0, 12))
print(f"model 1:{a_model_1.fit().aic}")
a_model_2 = SARIMAX(log, order=(1, 1, 0), seasonal_order=(0, 1, 0, 12))
print(f"model 2:{a_model_2.fit().aic}")
a_model_3 = SARIMAX(log, order=(2, 1, 0), seasonal_order=(0, 1, 0, 12))
print(f"model 3:{a_model_3.fit().aic}")

# ############################# EJ2 #############################

b_model_1 = SARIMAX(
    data[f"log({variable_name})"],
    order=(2, 1, 0),
    seasonal_order=(1, 1, 0, 12),
    exog=data[["LY", "TD1", "TD2"]],
)
residues = b_model_1.fit().resid
print(b_model_1.fit().summary())
############################# EJ4 #############################

residues_plot(residues, data["obs"], variable_name="Tiempo", save_path=save_path)

autocorrelation_plots(
    series=residues,
    lags=20,
    variable_name="Residuos",
    save_path=save_path,
)
# Q-Q plot for residuals
plt.figure(figsize=(10, 6))
sm.qqplot(residues, line="45", fit=True)
plt.title("Q-Q Plot of Residuals")
plt.savefig(os.path.join(save_path, "qqplot.png"))

check_white_noise(residues, exog=np.arange(1, len(residues) + 1))
