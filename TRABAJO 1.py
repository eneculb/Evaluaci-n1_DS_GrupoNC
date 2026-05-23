# INTEGRANTES:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print(" ANÁLISIS DE GENERACIÓN DE VIAJES Y TELETRABAJO")
print(" Fuente de datos: Simulación basada en Reporte de Movilidad 2025")
print("="*80)

# 1. GENERACIÓN Y LIMPIEZA DE DATOS

print("\n[1] GENERACIÓN Y LIMPIEZA DE DATOS")
print("-"*80)

np.random.seed(42)
n_usuarios = 500

data_viajes = {
    'es_hombre': np.random.randint(0, 2, n_usuarios), 'dias_teletrabajo': np.random.choice([0, 1, 2, 3, 4, 5], n_usuarios,  p=[0.6, 0.1, 0.1, 0.05, 0.05, 0.1]), 'ingreso_perc': np.random.uniform(1, 5, n_usuarios), 'edad': np.random.randint(18, 70, n_usuarios), 'tiene_auto': np.random.randint(0, 2, n_usuarios)
}

# Generar variable objetivo
data_viajes['viajes_diarios'] = (
    2.67 + 
    0.49 * data_viajes['es_hombre'] - 
    0.35 * data_viajes['dias_teletrabajo'] + 
    0.1 * data_viajes['ingreso_perc'] +
    0.15 * data_viajes['tiene_auto'] +
    np.random.normal(0, 0.4, n_usuarios)
)

# LIMPIEZA DE DATOS
print("Datos antes de limpieza:")
print(f"  - Registros totales: {n_usuarios}")
print(f"  - Valores negativos en viajes: {sum(data_viajes['viajes_diarios'] < 0)}")

# No pueden existir viajes negativos
data_viajes['viajes_diarios'] = np.maximum(0, data_viajes['viajes_diarios'])

df_generacion = pd.DataFrame(data_viajes)

# Eliminar outliers (valores > 3 desviaciones estándar)
mean_viajes = df_generacion['viajes_diarios'].mean()
std_viajes = df_generacion['viajes_diarios'].std()
df_generacion = df_generacion[
    (df_generacion['viajes_diarios'] >= mean_viajes - 3*std_viajes) & 
    (df_generacion['viajes_diarios'] <= mean_viajes + 3*std_viajes)
]

print(f"\nDatos después de limpieza:")
print(f"  - Registros finales: {len(df_generacion)}")
print(f"  - Registros eliminados: {n_usuarios - len(df_generacion)}")
print(f"  - Valores nulos: {df_generacion.isnull().sum().sum()}")







