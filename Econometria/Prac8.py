"""
1. Represent the time series and find the classical components(tendency, stationality and seasonality)
2. Determine the extraseasonal component, the seasonal component and the irregular component.
3. Obtain the Autocorrelation Function and the Partial Autocorrelation Function.
4. Diferentiatie in the adequate order.
"""

import pandas as pd
import seaborn as sns


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


from matplotlib import pyplot as plt

from utilities import (
    check_stationarity,
    suggest_arima_parameters,
    make_series_stationary,
    suggest_sarima_parameters,
)


# Significance level
threshold = 0.05

# Load the CSV file
file_path = "Econometria/MST050.csv"
data = pd.read_csv(file_path)
print(data.head())

######################################### GRAPHS #########################################
print(
    "",
    "\n------------------------------Graph Representation------------------------------",
    "\n",
)
y = data["PERSONAL"]
data["obs"] = pd.to_datetime(data["obs"], format="%YM%m")


plt.figure(figsize=(10, 6))
plt.plot(
    data["obs"],
    y,
    marker="o",
    linestyle="-",
)  # Line plot with points
plt.axhline(y=0, color="r", linestyle="--")
plt.title("Time Series Plot of PERSONAL Data")
plt.xlabel("Date")
plt.ylabel("PERSONAL")
plt.grid(True)
plt.ylim(bottom=y.min())
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Grouping the data by year
data["Year"] = data["obs"].dt.year
grouped_data = data.groupby("Year").agg(["mean", "max", "min"])

print(grouped_data)

# Calculating the range for each year
grouped_data["Range"] = (
    grouped_data["PERSONAL"]["max"] - grouped_data["PERSONAL"]["min"]
)

# Preparing data for plotting
mean_values = grouped_data["PERSONAL"]["mean"]
range_values = grouped_data["Range"]

# Plotting the range mean graph
plt.figure(figsize=(10, 6))
plt.scatter(mean_values, range_values)
plt.title("Range Mean Graph by Year")
plt.xlabel("Mean of PERSONAL")
plt.ylabel("Range of PERSONAL")
plt.grid(True)

# Show the plot
plt.show()

# ----------------------------- ANALYSIS OF THE SEASONAL COMPONENT -----------------------------#
# Extracting month and year from the date
data["Month"] = data["obs"].dt.month

# Creating a pivot table for the annual subseries plot
pivot_data = data.pivot_table(
    values="PERSONAL", index="Month", columns="Year", aggfunc="mean"
)

# Plotting the annual subseries
plt.figure(figsize=(12, 8))
sns.lineplot(data=pivot_data, dashes=False)
plt.title("Annual Subseries Plot of PERSONAL Data")
plt.xlabel("Month")
plt.ylabel("PERSONAL")
plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()

# We are using an additive model because the seasonal variation is constant over time.
decomposition = seasonal_decompose(data["PERSONAL"], model="multiplicatibe", period=12)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plotting the components
plt.figure(figsize=(14, 8))

# Plot for the trend component
plt.subplot(411)
plt.plot(data["obs"], data["PERSONAL"], label="Original")
plt.legend(loc="best")
plt.title("Original Time Series")
plt.grid(True)
plt.xlim(data["obs"][0])
plt.xticks(rotation=45)
plt.tight_layout()

# Plot for the trend component
plt.subplot(412)
plt.plot(data["obs"], trend, label="Trend")
plt.legend(loc="best")
plt.grid(True)
plt.xlim(data["obs"][0])

plt.xticks(rotation=45)
plt.tight_layout()

# Plot for the seasonal component
plt.subplot(413)
plt.plot(data["obs"], seasonal, label="Seasonality")
plt.legend(loc="best")
plt.grid(True)
plt.xlim(data["obs"][0])

plt.xticks(rotation=45)
plt.tight_layout()

# Plot for the residual component
plt.subplot(414)
plt.plot(data["obs"], residual, label="Residuals")
plt.legend(loc="best")
plt.grid(True)
plt.xlim(data["obs"][0])

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

######################################### Correlation and Autocorrelation #########################################

stationary_series, num_differences = make_series_stationary(data["PERSONAL"])
print(f" \n Number of differences applied: {num_differences} \n")
check = check_stationarity(stationary_series)


print(f"Analysis of stationarity: {check}\n")
# Gráficos ACF y PACF
print(f"length: {len(stationary_series)}")
lags = 24

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(stationary_series, ax=ax1, lags=lags)
plot_pacf(stationary_series, ax=ax2, lags=lags)

plt.tight_layout()
plt.show()
######################################### ARIMA #########################################

# Obtén los valores de ACF y PACF
acf_vals = acf(stationary_series, nlags=lags)
pacf_vals = pacf(stationary_series, nlags=lags)

# Suponiendo un intervalo de confianza del 95%
confidence_level = 1.96

# Sugerir valores de p y q
p, q = suggest_arima_parameters(acf_vals, pacf_vals, confidence_level)


print("\n----------------------ARIMA Model ----------------------\n")
model = ARIMA(stationary_series, order=(p, num_differences, q))
results = model.fit()
print(results.summary())

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
# plot_acf(model, ax=ax1, lags=lags)
# plot_pacf(model, ax=ax2, lags=lags)

plt.tight_layout()
plt.show()

print("\n----------------------SARIMA Model ----------------------\n")
p, q, P, D, Q = suggest_sarima_parameters(
    acf_vals, pacf_vals, num_differences, 12, confidence_level
)
model = ARIMA(
    stationary_series, order=(p, num_differences, q), seasonal_order=(P, D, Q, 12)
)
results = model.fit()
print(results.summary())

print("\n----------------------SARIMAX Model ----------------------\n")
model = SARIMAX(
    stationary_series, order=(p, num_differences, q), seasonal_order=(P, D, Q, 12)
)
results = model.fit()
print(results.summary())
