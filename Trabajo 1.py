import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

#IMPACTO DEL TELETRABAJO EN LA TASA DE VIAJES
print(" ANÁLISIS DE GENERACIÓN DE VIAJES Y TELETRABAJO")

# Simulación basada en datos del Reporte 2025
np.random.seed(42)
n_usuarios = 500

data_viajes = {
    # 0 = Femenino (Tasa promedio 2.67), 1 = Masculino (Tasa promedio 3.16)
    'es_hombre': np.random.randint(0, 2, n_usuarios),
    # Días de teletrabajo a la semana (0 a 5)
    'dias_teletrabajo': np.random.choice([0, 1, 2, 3, 4, 5], n_usuarios, p=[0.6, 0.1, 0.1, 0.05, 0.05, 0.1]),
    # Nivel de ingresos simulado 
    'ingreso_perc': np.random.uniform(1, 5, n_usuarios) 
}

# Generar variable objetivo: Tasa de viajes diarios 
data_viajes['viajes_diarios'] = (
    2.67 + 
    0.49 * data_viajes['es_hombre'] - 
    0.35 * data_viajes['dias_teletrabajo'] + 
    0.1 * data_viajes['ingreso_perc'] +
    np.random.normal(0, 0.4, n_usuarios) # Ruido aleatorio
)

# Limpiar datos: No pueden existir viajes negativos
data_viajes['viajes_diarios'] = np.maximum(0, data_viajes['viajes_diarios']) 

df_generacion = pd.DataFrame(data_viajes)
print(f"Promedio global simulado: {df_generacion['viajes_diarios'].mean():.2f} viajes/día (Cercano al 3.07 del reporte)\n")

# División de datos y entrenamiento del Modelo
X_gen = df_generacion[['es_hombre', 'dias_teletrabajo', 'ingreso_perc']]
y_gen = df_generacion['viajes_diarios']

X_train_gen, X_test_gen, y_train_gen, y_test_gen = train_test_split(X_gen, y_gen, test_size=0.2, random_state=42)

modelo_generacion = LinearRegression()
modelo_generacion.fit(X_train_gen, y_train_gen)

# Resultados
print(f"Intercepto (β₀): {modelo_generacion.intercept_:.2f}")
coeficientes_df = pd.DataFrame({
    'Variable': X_gen.columns,
    'Coeficiente': modelo_generacion.coef_
}).sort_values('Coeficiente', ascending=False)

print("\nImpacto de cada variable en los viajes diarios:")
print(coeficientes_df.to_string(index=False))
print("\n")

# PREDICCIÓN DE TIEMPOS DE VIAJE POR MODO Y DISTANCIA
print(" MODELO DE TIEMPOS DE VIAJE POR MODO")

# Simulación de matriz de viajes (Basado en distancias del Reporte)
np.random.seed(42)
modos = ['Bus', 'Metro', 'Auto', 'Caminata']
df_tiempos = pd.DataFrame({
    'distancia_km': np.random.uniform(1, 15, 300),
    'modo': np.random.choice(modos, 300, p=[0.27, 0.15, 0.28, 0.30]) # Partición modal
})

# Generar tiempo real simulado en minutos usando velocidades promedio
# (Velocidad Auto ~25km/h, Metro ~30km/h, Bus ~18km/h, Caminata ~4.5km/h)
velocidades = {'Auto': 25/60, 'Metro': 30/60, 'Bus': 18/60, 'Caminata': 4.5/60}
df_tiempos['tiempo_min'] = df_tiempos.apply(
    lambda row: row['distancia_km'] / velocidades[row['modo']] + np.random.normal(0, 5), axis=1
)

# One-Hot Encoding: Convertir el modo categórico a variables Dummy
df_tiempos_encoded = pd.get_dummies(df_tiempos, columns=['modo'], drop_first=True)

# Entrenamiento del Modelo
X_time = df_tiempos_encoded.drop('tiempo_min', axis=1)
y_time = df_tiempos_encoded['tiempo_min']

modelo_tiempos = LinearRegression()
modelo_tiempos.fit(X_time, y_time)

# Evaluación (MAE y R²)
r2_tiempos = modelo_tiempos.score(X_time, y_time)
y_pred_tiempos = modelo_tiempos.predict(X_time)
mae_tiempos = mean_absolute_error(y_time, y_pred_tiempos)

print(f"R² del modelo: {r2_tiempos:.4f}")
print(f"Error Absoluto Medio (MAE): {mae_tiempos:.2f} minutos.")
print("\nInterpretación:")
print("Las predicciones de tiempo de llegada de este modelo fallan, en promedio, por ese margen de minutos.")
print("="*60)