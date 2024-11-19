import pandas as pd
import xml.etree.ElementTree as ET
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_breuschpagan

from utilities import compare_models


# Leer y procesar el archivo XML
with open("/Users/carlosedm10/projects/college/Econometria/MRL151-1.gdt", "r") as file:
    lines = file.readlines()
root = ET.fromstring("".join(lines))
data_dict = {"CANTIDAD": [], "PERDIDAS": []}
for obs in root.findall(".//obs"):
    label = obs.get("label")
    values = obs.text.split()
    if len(values) == 2:
        cantidad, perdidas = map(float, values)
        data_dict["CANTIDAD"].append(cantidad)
        data_dict["PERDIDAS"].append(perdidas)
data_df = pd.DataFrame(data_dict)

# Ajustar el modelo lineal
data_df["CANTIDAD_AJUSTADA"] = data_df["CANTIDAD"] - 100000
X = sm.add_constant(data_df["CANTIDAD_AJUSTADA"])
y = data_df["PERDIDAS"]
model = compare_models(X, y)
residuos = model.resid
valores_ajustados = model.fittedvalues

# Gráfica de residuos vs. valores ajustados
plt.scatter(valores_ajustados, residuos, color="blue")
plt.axhline(y=0, color="red", linestyle="--")
plt.title("Gráfico de Residuos vs. Valores Ajustados")
plt.xlabel("Valores Ajustados")
plt.ylabel("Residuos")
plt.grid(True)
plt.show()

# Prueba de Breusch-Pagan para la heterocedasticidad

bp_test = het_breuschpagan(residuos, X)
print("Resultados de la prueba de Breusch-Pagan:", bp_test)

# Estimar el modelo con errores estándar robustos
robust_model = sm.OLS(y, X).fit(cov_type="HC3")
print(robust_model.summary())

# Estimar la varianza de los residuos para WLS
data_df["RESIDUOS_AL_CUADRADO"] = residuos**2
X_variance = sm.add_constant(data_df["CANTIDAD_AJUSTADA"])
y_variance = data_df["RESIDUOS_AL_CUADRADO"]
model_variance = sm.OLS(y_variance, X_variance).fit()
data_df["ESTIMACION_VARIANZA"] = model_variance.fittedvalues

# Ajustar un modelo WLS
data_df["PONDERACIONES"] = 1 / data_df["ESTIMACION_VARIANZA"]
wls_model = sm.WLS(y, X, weights=data_df["PONDERACIONES"]).fit()
print(wls_model.summary())
