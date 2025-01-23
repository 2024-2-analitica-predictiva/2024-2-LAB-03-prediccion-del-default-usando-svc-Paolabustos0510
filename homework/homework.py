# %% [markdown]
# # Laboratorio 03
# 

# %% [markdown]
# ## Importar librerías

# %%
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

import pickle
import gzip
import json


# %% [markdown]
# ## Paso 1: Carga y Limpieza de Datos

# %%
#agrupar educaciones mayores que 4 (Others)
def agrupar_educaciones(codigo_educacion):
    if codigo_educacion > 4:
        return 4
    return codigo_educacion


def cargar_limpiar_dataset(nombre_archivo):
    datos = pd.read_csv(nombre_archivo)
    
    # - Renombre la columna "default payment next month" a "default".
    datos = datos.rename(columns={"default payment next month" : "default"})

    # Remueva la columna "ID".
    datos.drop(columns=["ID"], inplace = True)

    # Elimine los registros con informacion no disponible.
    datos =  datos[(datos['EDUCATION'] != 0) & (datos['MARRIAGE'] != 0)]

    # Para la columna EDUCATION, valores > 4 indican niveles superiores de educación, agrupe estos valores en la categoría "others"
    datos["EDUCATION"] = datos["EDUCATION"].apply(agrupar_educaciones)

    return datos

datos_entrenamiento = cargar_limpiar_dataset("files/input/train_default_of_credit_card_clients.csv")
datos_prueba = cargar_limpiar_dataset("files/input/test_default_of_credit_card_clients.csv")

print("Datos de Entrenamiento")
print(datos_entrenamiento.describe(include='all'))

print("Datos de Prueba")
print(datos_prueba.describe(include='all'))

# %% [markdown]
# ## Paso 2: Divida los datasets en x_train, y_train, x_test, y_test

# %%
x_train = datos_entrenamiento.drop(columns=["default"])
y_train = datos_entrenamiento["default"]

x_test = datos_prueba.drop(columns=["default"])
y_test = datos_prueba["default"]

# %% [markdown]
# ## Paso 3: Pipeline

# %%
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).

# Listas de variables
variables_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
variables_numericas = [col for col in x_train.columns if col not in variables_categoricas]

# Transformaciones
transformer_variables_categoricas = OneHotEncoder()
transformer_variables_numericas = StandardScaler()

# preprocesador
preprocesador = ColumnTransformer(
    transformers=[
        ("categoricas", transformer_variables_categoricas, variables_categoricas),
        ("numericas",   transformer_variables_numericas,   variables_numericas  ),
    ],
)

# Componentes principales
pca = PCA()

# Elegir k mejores características
k_mejores = SelectKBest(score_func=f_classif)

# Crear modelo de regresión logística
modelo = SVC()

# Pipeline completo
pipeline = Pipeline(
    steps=[
        ("preprocesador", preprocesador), 
        ("pca", pca),
        ("kmejores", k_mejores),
        ("SVC", modelo)
    ]
)

# %% [markdown]
# # Paso 4: Optimización de Hiperparámetros

# %%
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.


# Establecer hiperparámetros a evaluar
param_grid = [
    {
        "kmejores__k"         : [12],
        'pca__n_components'   : [20],
        'SVC__gamma'          : [0.1],
    }
]

# Creación malla de hiperpárametros
busqueda_malla = GridSearchCV(
    estimator=pipeline, 
    param_grid=param_grid,
    scoring="balanced_accuracy",
    cv=10,
    verbose=3,
    # n_jobs=2,
)

# Entrenamiento de modelo
busqueda_malla.fit(x_train, y_train)

# %%
# Guardar el mejor modelo

mejor_modelo = busqueda_malla.best_estimator_
mejores_parametros = busqueda_malla.best_params_
mejor_resultado = busqueda_malla.best_score_
print("Parámetros encontrados: ", mejores_parametros)
print("Mejor resultado: ", mejor_resultado)

# %% [markdown]
# # Paso 5: Guardar Modelo

# %%
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
with gzip.open("files/models/model.pkl.gz", "wb") as archivo:
    pickle.dump(busqueda_malla, archivo)

# %% [markdown]
# # Paso 6 y 7: Cálculo Métricas y matriz de confusión

# %%
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba.

# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba

# Función para calcular métricas
def calcular_metricas(modelo, x, y, tipo_dataset): 
    y_pred = modelo.predict(x) 
    diccionario_metricas = {
        "type" : "metrics",
        "dataset" : tipo_dataset,
        "precision" : float(precision_score(y, y_pred, zero_division=0)),
        "balanced_accuracy" : float(balanced_accuracy_score(y, y_pred)), 
        "recall" : float(recall_score(y, y_pred)), 
        "f1_score" : float(f1_score(y, y_pred)), 
    }

    return diccionario_metricas

# Función para calcular matriz de confusión
def calcular_matriz_confusion(modelo, x, y, tipo_dataset):
    matriz_con = confusion_matrix(y, modelo.predict(x))
    diccionario_matriz = {
        "type": "cm_matrix",
        "dataset": tipo_dataset,
        "true_0": {"predicted_0": int(matriz_con[0, 0]), "predicted_1": int(matriz_con[0, 1])},
        "true_1": {"predicted_0": int(matriz_con[1, 0]), "predicted_1": int(matriz_con[1, 1])},
    }
    return diccionario_matriz

# Guardar métricas y matrices de consufión
valores = [
    calcular_metricas(mejor_modelo, x_train, y_train, "train"),
    calcular_metricas(mejor_modelo, x_test, y_test, "test"),
    calcular_matriz_confusion(mejor_modelo, x_train, y_train,"train"),
    calcular_matriz_confusion(mejor_modelo, x_test, y_test, "test"),
]


# %%

# Revisar valores
print(valores)

# Guardar archivo JSON
with open("files/output/metrics.json", "w") as archivo:
    for v in valores:
        json.dump(v, archivo)
        archivo.write("\n")


