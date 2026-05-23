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

# 2. ESTADÍSTICA DESCRIPTIVA

print("\n[2] ESTADÍSTICA DESCRIPTIVA")
print("-"*80)

# Medidas de tendencia central y dispersión
print("\n VIAJES DIARIOS:")
print(f"  • Promedio (μ): {df_generacion['viajes_diarios'].mean():.3f}")
print(f"  • Mediana: {df_generacion['viajes_diarios'].median():.3f}")
print(f"  • Moda: {df_generacion['viajes_diarios'].mode()[0]:.3f}")
print(f"  • Desviación Estándar (σ): {df_generacion['viajes_diarios'].std():.3f}")
print(f"  • Varianza (σ²): {df_generacion['viajes_diarios'].var():.3f}")
print(f"  • Coeficiente de Variación: {(df_generacion['viajes_diarios'].std()/df_generacion['viajes_diarios'].mean())*100:.2f}%")
print(f"  • Rango: {df_generacion['viajes_diarios'].max() - df_generacion['viajes_diarios'].min():.3f}")
print(f"  • Mínimo: {df_generacion['viajes_diarios'].min():.3f}")
print(f"  • Máximo: {df_generacion['viajes_diarios'].max():.3f}")

# Cuartiles
print(f"\n Cuartiles:")
print(f"    - Q1 (25%): {df_generacion['viajes_diarios'].quantile(0.25):.3f}")
print(f"    - Q2 (50%): {df_generacion['viajes_diarios'].quantile(0.50):.3f}")
print(f"    - Q3 (75%): {df_generacion['viajes_diarios'].quantile(0.75):.3f}")
print(f"    - IQR: {df_generacion['viajes_diarios'].quantile(0.75) - df_generacion['viajes_diarios'].quantile(0.25):.3f}")

# Asimetría y Curtosis
print(f"\n Forma de la distribución:")
print(f"    - Asimetría (Skewness): {df_generacion['viajes_diarios'].skew():.3f}")
print(f"    - Curtosis: {df_generacion['viajes_diarios'].kurtosis():.3f}")

# ESTADÍSTICAS POR GRUPO
print("\n COMPARACIÓN POR GÉNERO:")
print(df_generacion.groupby('es_hombre')['viajes_diarios'].agg([
    ('Promedio', 'mean'),
    ('Desv_Std', 'std'),
    ('Mínimo', 'min'),
    ('Máximo', 'max'),
    ('Conteo', 'count')
]).round(3))

print("\n COMPARACIÓN POR DÍAS DE TELETRABAJO:")
print(df_generacion.groupby('dias_teletrabajo')['viajes_diarios'].agg([
    ('Promedio', 'mean'),
    ('Desv_Std', 'std'),
    ('Conteo', 'count')
]).round(3))

# MATRIZ DE COVARIANZA Y CORRELACIÓN
print("\n MATRIZ DE COVARIANZA:")
cov_matrix = df_generacion[['viajes_diarios', 'dias_teletrabajo', 'ingreso_perc', 'edad']].cov()
print(cov_matrix.round(3))

print("\n MATRIZ DE CORRELACIÓN:")
corr_matrix = df_generacion[['viajes_diarios', 'dias_teletrabajo', 'ingreso_perc', 'edad', 'tiene_auto']].corr()
print(corr_matrix.round(3))

# Covarianza específica
cov_teletrabajo = df_generacion['viajes_diarios'].cov(df_generacion['dias_teletrabajo'])
print(f"\n  • Covarianza (Viajes vs Teletrabajo): {cov_teletrabajo:.3f}")
print(f"  • Correlación (Viajes vs Teletrabajo): {df_generacion['viajes_diarios'].corr(df_generacion['dias_teletrabajo']):.3f}")













