import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from IPython.display import display
from pmdarima.arima import ARIMA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_breuschpagan
from pandas.core.api import DataFrame
from itertools import product
from scipy import stats
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.seasonal import seasonal_decompose
from numpy.typing import ArrayLike


def compute_residuals(target_var, predictors, df):
    """Compute residuals for a specific target variable."""
    y = df[target_var]
    X = df[predictors]

    # Add a constant (intercept) to the independent variables
    X = sm.add_constant(X)

    # Fit the model
    model = sm.OLS(y, X).fit()

    # Return residuals
    return model.resid


def eliminate_max_correlated_with_target(df, target):
    """
    Removes the independent variable that has the highest absolute correlation with the target variable.

    Args:
    df (pd.DataFrame): Dataframe with the correlation matrix.
    target (str): Target variable name.

    Returns:
    pd.DataFrame: Updated dataframe with one less independent variable.
    """
    # Compute correlation of all independent variables with the target
    correlations = df.corr()[target].drop(target).abs()

    # Identify the variable with the highest absolute correlation with the target
    column_to_drop = correlations.idxmax()

    # Drop the identified column from the dataframe
    df = df.drop(columns=[column_to_drop])

    return df, column_to_drop


def eliminate_variable_high_pvalue(model, threshold=0.05):
    """
    Identify the variable with the highest p-value greater than the threshold.

    Args:
    model (RegressionResults): Fitted regression model from statsmodels.
    threshold (float, optional): Significance level threshold. Defaults to 0.05.

    Returns:
    str: Name of the variable to be eliminated. Returns None if no variables exceed the threshold.
    """
    # Exclude the constant term when checking p-values
    p_values = model.pvalues.drop("const", errors="ignore")

    # Filter variables with p-values greater than the threshold
    high_p_values = p_values[p_values > threshold]

    # If no variable exceeds the threshold, return None
    if high_p_values.empty:
        return None

    # Identify and return the variable with the highest p-value
    return high_p_values.idxmax()


def backward_elimination(X, y, threshold):
    """
    Realiza la eliminación progresiva para un modelo lineal.

    Parámetros:
    - X: matriz de predictores.
    - y: vector de la variable de respuesta.
    - threshold: umbral de significancia para mantener una variable en el modelo.

    Retorna:
    - Modelo final después de la eliminación progresiva.
    """

    num_vars = X.shape[1]
    for i in range(0, num_vars):
        model = sm.OLS(y, X).fit()
        max_p_value = max(model.pvalues)
        if max_p_value > threshold:
            remove = (
                model.pvalues.idxmax()
            )  # Identificar variable con el valor p más alto
            print("")
            print("Deleting {} with p-value {}".format(remove, max_p_value))
            X = X.drop(remove, axis=1)  # Eliminar variable
        else:
            model = sm.OLS(y, sm.add_constant(X)).fit()
            break  # TODO: change this to return model

    return model


# Uso del algoritmo:
# X es tu dataframe de predictores y y es tu serie/vector de respuesta.
# final_model = backward_elimination(X, y)


def forward_selection(X, y):
    """
    Performs forward selection based on residual analysis.

    Parameters:
    - X: DataFrame of predictors.
    - y: Series/vector of response variable.

    Returns:
    - Final model after forward selection.
    """
    remaining_predictors = list(X.columns)
    included_predictors = []
    current_score, best_new_score = float("inf"), float(
        "inf"
    )  # initialized with infinity

    while remaining_predictors and current_score == best_new_score:
        scores_with_predictors = []
        for predictor in remaining_predictors:
            formula = "{} ~ {}".format(
                y.name, " + ".join(included_predictors + [predictor])
            )
            score = sm.OLS.from_formula(formula, data=X.join(y)).fit().ssr
            scores_with_predictors.append((score, predictor))

        scores_with_predictors.sort(reverse=True)
        best_new_score, best_predictor = scores_with_predictors.pop()
        if current_score > best_new_score:
            included_predictors.append(best_predictor)
            remaining_predictors.remove(best_predictor)
            current_score = best_new_score

    formula = "{} ~ {}".format(y.name, " + ".join(included_predictors))
    model = sm.OLS.from_formula(formula, data=X.join(y)).fit()

    return model


# Using the algorithm:
# X is your DataFrame of predictors and y is your Series/vector of response.
# final_model = forward_selection(X, y)


def calculate_max_vif(X):
    """Calculate the maximum VIF for the predictors in X."""
    vif_data = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return max(vif_data)


def compare_models(X, y):
    # Apply both methods to obtain models
    model_backward = backward_elimination(X, y, 0.05)
    model_forward = forward_selection(X, y)

    # Compare R^2 adjusted
    if model_backward.rsquared_adj > model_forward.rsquared_adj:
        print("backward_elimination")
        return model_backward
    else:
        print("forward_selection")
        return model_forward


def create_dummies(data: DataFrame) -> DataFrame:
    """
    Create all possible dummy variables for categorical variables.
    The created dummy variables heve the format: {value}
    For example for the column "NPROV" with possible values "ALICANTE", "CASTELLÓN" and "VALENCIA",
    the dummy variables will be "ALICANTE", "CASTELLÓN" and "VALENCIA".
    """
    # Get the categorical variables columns
    categorical_columns = data.select_dtypes(include=["object"]).columns

    # Create dummy variables
    dummies = pd.get_dummies(
        data[categorical_columns],
        prefix="",
        prefix_sep="",
        dtype=int,
    )
    # Concatenate the dummy variables with the original data
    data = pd.concat([data, dummies], axis=1)
    return data


def create_interactions(data: DataFrame) -> DataFrame:
    """
    Create all possible interactions between variables.
    """
    # Iterate over all possible combinations of columns
    for col1, col2 in product(data.columns, data.columns):
        # If the column is categorical, skip it
        categorical_columns = data.select_dtypes(include=["object"]).columns
        if col1 in categorical_columns or col2 in categorical_columns:
            continue

        # If the column is not the same, create the interaction variable
        if col1 != col2:
            data[f"{col1}_{col2}"] = data[col1] * data[col2]

    return data


def make_series_stationary(
    series, max_diff=3, p_value_threshold=0.05, seasonal_period=12
):
    """
    Apply differencing to a time series until it becomes stationary.

    :param series: The original time series.
    :param max_diff: Maximum number of differencing allowed.
    :param p_value_threshold: Threshold for the p-value to consider the series stationary.
    :return: A tuple containing the differenced series and the number of differences applied.
    """

    def adf_test(serie):
        result = adfuller(serie, autolag="AIC", regression="ct")
        return result[1]  # p-value

    # Initial ADF test
    p_value = adf_test(series)

    num_diff = 0
    num_seasonal_diff = 0

    # Apply differencing until stationary or max_diff reached
    if p_value < 0.05:
        print("Series is already stationary.")

    while p_value > p_value_threshold and num_diff < max_diff:
        num_diff += 1
        series = series.diff().dropna()
        p_value = adf_test(series)
        print(f"ADF test p-value: {p_value}")

    # Apply seasonal differencing until stationary or max_diff reached
    while p_value > p_value_threshold and num_seasonal_diff < max_diff:
        num_seasonal_diff += 1
        series = series.diff(periods=seasonal_period).dropna()
        p_value = adf_test(series)

    return series, num_diff, num_seasonal_diff


# Chequeo de estacionariedad
def check_stationarity(series):
    result = adfuller(series)
    print("Estadístico ADF:", result[0])
    print("Valor p:", result[1])
    print("Valores críticos:")
    for key, value in result[4].items():
        print(f"    {key}: {value}")


def suggest_arima_parameters(acf_values, pacf_values, confidence_interval):
    """
    Suggest ARIMA parameters p and q based on ACF and PACF values.

    :param acf_values: Array of ACF values.
    :param pacf_values: Array of PACF values.
    :param confidence_interval: Confidence interval (e.g., 1.96 for 95%).
    :return: Tuple (p, q) as suggested parameters.
    """
    p = sum(abs(pacf_values) > confidence_interval)
    q = sum(abs(acf_values) > confidence_interval)

    return p, q


def suggest_sarima_parameters(acf_values, pacf_values, s, confidence_interval):
    """
    Suggest SARIMA parameters (p, d, q, P, D, Q) based on ACF and PACF values.

    :param acf_values: Array of ACF values.
    :param pacf_values: Array of PACF values.
    :param s: Seasonal period.
    :param confidence_interval: Confidence interval (e.g., 1.96 for 95%).
    :return: Tuple (p, d, q, P, D, Q) as suggested parameters.
    """
    # Non-seasonal p and q
    p = sum(abs(pacf_values[: s - 1]) > confidence_interval)
    q = sum(abs(acf_values[: s - 1]) > confidence_interval)

    # Seasonal P and Q
    P = sum(abs(pacf_values[s - 1 :: s]) > confidence_interval)
    Q = sum(abs(acf_values[s - 1 :: s]) > confidence_interval)

    # Assuming D=1 as a common practice for seasonal differencing

    return p, q, P, Q


def format_models(models):
    # Formatting the output to match the GRETL style
    model_strings = []
    for model in models:
        # Assuming the model tuple structure is (params, AIC, BIC)
        params, aic, bic = model
        model_str = f"({' ,'.join(map(str, params[:-1]))}){params[-1]} - AIC: {aic:.5f} - BIC: {bic:.5f}"
        model_strings.append(model_str)
    return model_strings


def best_arima(time_series, d, D, max_p=3, max_q=3):
    best_models_aic = []
    best_models_bic = []

    # Iterate over various combinations of p, q, P, and Q
    for p in range(max_p):
        for q in range(max_q):
            for P in range(max_p):
                for Q in range(max_q):
                    # Fit the ARIMA model
                    model = ARIMA(
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, 12),
                        suppress_warnings=True,
                    )
                    model_fit = model.fit(time_series)

                    # Append the model and its AIC/BIC to the lists
                    best_models_aic.append((model_fit.aic(), model))
                    best_models_bic.append((model_fit.bic(), model))

    # Sort the models by AIC and BIC
    best_models_aic.sort(key=lambda x: x[0])
    best_models_bic.sort(key=lambda x: x[0])

    # Return the top 5 models based on AIC and BIC
    return best_models_aic[:5], best_models_bic[:5]


def check_white_noise(residuals, exog, alpha=0.05):
    """
    Check if the residuals are white noise using the following tests:
    - Mean Value Test
    - Heteroscedasticity Tests
        - White Test
        - Breusch-Pagan Test
        - F and t tests
    - Normality Tests
        - Shapiro-Wilk Test
        - Jarque-Bera Test
    - Autocorrelation Test (Durbin-Watson Test)

    Args:
    residuals (pd.Series): Residuals of the model.
    alpha (float, optional): Significance level. Defaults to 0.05.

    Returns:
    dict: Dictionary with the results of the tests.
    """

    def format_diagnostics(diagnostics):
        print("\nDiagnostic Test Results:")
        print("-" * 50)
        for key, value in diagnostics.items():
            print(f"{key.ljust(10)}: {value}")

    # Defining the model:
    squared_residuals = residuals**2
    y = exog
    t = np.arange(1, len(y) + 1)
    diagnostics = {}

    all_tests_passed = True

    # 1. Mean Value Test
    _, p_value_mean = stats.ttest_1samp(residuals, 0)
    diagnostics["Mean Test p-value"] = p_value_mean
    diagnostics["Mean Test"] = "Pass" if p_value_mean > alpha else "Fail"
    if p_value_mean <= alpha:
        all_tests_passed = False

    # 2. Heteroscedasticity Tests
    # White Test
    _, p_value_white, _, _ = het_white(squared_residuals, sm.add_constant(y))
    diagnostics["White Test p-value"] = p_value_white
    diagnostics["White Test"] = "Pass" if p_value_white > alpha else "Fail"
    if p_value_white <= alpha:
        all_tests_passed = False

    # Breusch-Pagan Test
    _, _, _, p_value_breusch_pagan = het_breuschpagan(residuals, sm.add_constant(y))
    diagnostics["Breusch-Pagan Test p-value"] = p_value_breusch_pagan
    diagnostics["Breusch-Pagan Test"] = (
        "Pass" if p_value_breusch_pagan > alpha else "Fail"
    )
    if p_value_breusch_pagan <= alpha:
        all_tests_passed = False

    # F and t tests
    residual_linear_model = sm.OLS(squared_residuals, t).fit()
    f_pvalue = residual_linear_model.f_pvalue  # P-value for the F-statistic
    diagnostics["F Test p-value"] = f_pvalue
    diagnostics["F Test"] = "Pass" if f_pvalue > alpha else "Fail"
    if f_pvalue > alpha:
        all_tests_passed = False

    # 3. Normality Tests
    # Add test Kolmogorov-Smirnov
    _, p_value = stats.kstest(residuals, "norm")
    diagnostics["Kolmogorov-Smirnov Test p-value"] = p_value
    diagnostics["Kolmogorov-Smirnov Test"] = "Pass" if p_value > alpha else "Fail"
    if p_value <= alpha:
        all_tests_passed = False

    # Shapiro-Wilk Test
    _, p_value_shapiro = stats.shapiro(residuals)
    diagnostics["Shapiro Test p-value"] = p_value_shapiro
    diagnostics["Shapiro Test"] = "Pass" if p_value_shapiro > alpha else "Fail"
    if p_value_shapiro <= alpha:
        all_tests_passed = False

    # Jarque-Bera Test
    _, p_value_jarque_bera, _, _ = jarque_bera(residuals)
    diagnostics["Jarque-Bera Test p-value"] = p_value_jarque_bera
    diagnostics["Jarque-Bera Test"] = "Pass" if p_value_jarque_bera > alpha else "Fail"
    if p_value_jarque_bera <= alpha:
        all_tests_passed = False

    # 4. Autocorrelation Test (Durbin-Watson Test)
    dw_stat = durbin_watson(residuals)
    diagnostics["Durbin-Watson stat"] = dw_stat
    # Interpret Durbin-Watson statistic
    if dw_stat < 1.5 or dw_stat > 2.5:
        diagnostics["Durbin-Watson"] = "Fail"
        all_tests_passed = False
    else:
        diagnostics["Durbin-Watson"] = "Pass"

    # Final Verdict
    diagnostics["Final Verdict"] = (
        "The Residues are White Noise"
        if all_tests_passed
        else "The Residues are not White Noise"
    )

    return format_diagnostics(diagnostics)


def format_diagnostics(diagnostics):
    print("\nDiagnostic Test Results:")
    print("-" * 50)
    for key, value in diagnostics.items():
        print(f"{key.ljust(10)}: {value}")


############################### GRAPHS ###############################


def time_plot(
    *args: ArrayLike,
    variable_name: str,
    xlabel="Date",
    ylim=None,
    save_path=None,
):
    """
    This function plots a time series plot with the following characteristics:
    - Line plot with points
    - Horizontal line at y=0
    - Title with the name of the variable
    - X and Y axis labels
    - Grid
    - Y axis starts at 0
    - X axis labels rotated 45 degrees

    Args:
    x (pd.Series): X axis values.
    y (pd.Series): Y axis values.
    variable_name (str, optional): Name of the variable to be displayed in the title. Defaults to str.

    xlabel (str, optional): X axis label. Defaults to "Date".
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        *args,
        marker="o",
        linestyle="-",
        markersize=1,
    )  # Line plot with points
    plt.axhline(y=0, color="r", linestyle="--")
    plt.title(f"Time Series Plot of {variable_name} Data")
    plt.xlabel(xlabel)
    plt.ylabel(variable_name)
    plt.grid(True)
    plt.ylim(bottom=ylim)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path == None:
        plt.show()
    else:
        plt.savefig(f"{save_path}/{variable_name} time plot.png")


def plots_for_seasonality(data, variable_name: str, save_path=None):
    """
    This function plots a range mean graph with the following characteristics:
    - Scatter plot with points
    - Linear regression line
    - Title with the name of the variable
    - X and Y axis labels
    - Grid

    Args:
    data (pd.DataFrame): Dataframe with the data to be plotted.
    variable_name (str): Name of the variable to be displayed in the title. Defaults to str.
    """
    # Get the "Downloads" directory path
    downloads_directory = os.path.join(os.path.expanduser("~"), "Downloads")
    try:
        # Grouping the data by year
        data["Year"] = data["obs"].dt.year
        grouped_data = data.groupby("Year").agg(["mean", "max", "min"])

        # Calculating the range for each year
        grouped_data["Range"] = (
            grouped_data[variable_name]["max"] - grouped_data[variable_name]["min"]
        )

        # Preparing data for plotting
        mean_values = grouped_data[variable_name]["mean"]
        range_values = grouped_data["Range"]
        # Calculate the slope and intercept of the line
        slope, intercept = np.polyfit(mean_values, range_values, 1)

        # Create a figure with two subplots side by side
        plt.figure(figsize=(15, 6))

        # Subplot 1: Range Mean Graph by Year
        plt.subplot(1, 2, 1)
        plt.scatter(mean_values, range_values)
        plt.plot(mean_values, slope * mean_values + intercept, color="red")
        plt.title("Range Mean Graph by Year")
        plt.xlabel(f"Mean of {variable_name}")
        plt.ylabel(f"Range of {variable_name}")
        plt.grid(True)

        # Extracting month and year from the date
        data["Month"] = data["obs"].dt.month

        tittle = f"Series decomposition of {variable_name} data"

        # Subplot 2: Annual Subseries Plot
        plt.subplot(1, 2, 2)
        pivot_data = data.pivot_table(
            values=variable_name, index="Month", columns="Year", aggfunc="mean"
        )
        sns.lineplot(data=pivot_data, dashes=False)
        plt.title(f"Annual Subseries Plot of {variable_name} Data")
        plt.xlabel("Month")
        plt.ylabel(variable_name)
        plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True)

        plt.tight_layout()
        if save_path == None:
            plt.show()
        else:
            plt.savefig(f"{downloads_directory}/{tittle} plot.png")
    except Exception as e:
        print(f"Error: {e}")
        if variable_name not in data.columns:
            print(f"Variable {variable_name} not found in the data.")


def series_decomposition(data, variable_name: str):
    """
    This function plots different graphs for visualizing if the model is additive or multiplicative.
    It lets you decide which model to use, and then plots the decomposition of the series.

    args:
    data (pd.DataFrame): Dataframe with the data to be plotted.
    variable_name (str): Name of the variable to be displayed in the title. Obligatory.

    """
    # Get the "Downloads" directory path
    downloads_directory = os.path.join(os.path.expanduser("~"), "Downloads")
    plots_for_seasonality(data, variable_name)

    print("Given the data before, chose your model:")
    print("1. Additive")
    print("2. Multiplicative")

    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        # Additive: Append a phrase to the string
        model_type = choice
    elif choice == "2":
        # Multiplicative: Repeat the string 3 times
        model_type = choice
    else:
        # If the input is invalid, return the original string
        print("Invalid choice. Setting the default mode, additive.")
        model_type = "additive"

    # We are using an multiplicatibe model because the seasonal variation is constant over time.
    decomposition = seasonal_decompose(data[variable_name], model=model_type, period=12)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Plotting the components
    plt.figure(figsize=(14, 8))

    # Plot for the trend component
    plt.subplot(411)
    plt.plot(data["obs"], data[variable_name], label="Original")
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
    plt.savefig(f"{downloads_directory}/{variable_name} decomposition.png")


def autocorrelation_plots(series, variable_name: str, lags=12, save_path=None):
    """
    Plots the autocorrelation and partial autocorrelation functions for a time series.

    :param series: The time series.
    :param lags: Number of lags to plot.
    """
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(
        series,
        ax=ax1,
        lags=lags,
        title=f"Autocorrelation for {variable_name} time series",
    )
    plot_pacf(series, ax=ax2, lags=lags)

    plt.tight_layout()
    if save_path == None:
        plt.show()
    else:
        plt.savefig(f"{save_path}/{variable_name} autocorrelation.png")


def residues_plot(residues, variable, variable_name: str, save_path=None):
    """
    Args:
    residues (pd.Series): Residues of the model.
    variable (pd.Series): Independent variable.
    variable_name (str): Name of the independent variable.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x=variable, y=residues)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.title(f"{variable_name} vs Residues")
    plt.xlabel(variable_name)
    plt.ylabel("Residues")
    plt.tight_layout()
    if save_path == None:
        plt.show()
    else:
        plt.savefig(f"{save_path}/{variable_name} vs Residues.png")
