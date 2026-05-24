#parte 2— modelos predictivos modalidad urbana
# incluye: Regresión Lineal Múltiple, Árbol de Decisión y Random Forest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from datos import df_comunas, FEATURES, TARGET

# helper
def titulo(texto, nivel=1):
    if nivel == 1:
        linea = "="*60
        print(f"\n{linea}\n  {texto}\n{linea}")
    else:
        linea = "-" * 40
        print(f"\n{linea}\n  {texto}\n{linea}")

#evaluación de modelo
def evaluar(nombre, y_real, y_pred):
    mse  = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_real, y_pred)
    r2   = r2_score(y_real, y_pred)
    print(f"\n  [{nombre}]")
    print(f"    MSE   : {mse:.4f}")
    print(f"    RMSE  : {rmse:.4f}")
    print(f"    MAE   : {mae:.4f}")
    print(f"    R²    : {r2:.4f}  ({r2*100:.2f}% varianza explicada)")
    return {"nombre": nombre, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

def preparar_datos():
    X = df_comunas[FEATURES].values
    y = df_comunas[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    return X_train, X_test, y_train, y_test

#modelos
def regresion_lineal(X_train, X_test, y_train, y_test):
    titulo("2.1  REGRESIÓN LINEAL MÚLTIPLE", nivel=2)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    evaluar("Regresión Lineal — TRAIN", y_train, lr.predict(X_train))
    metricas = evaluar("Regresión Lineal — TEST",  y_test,  lr.predict(X_test))

    print("\n  Coeficientes:")
    for feat, coef in zip(FEATURES, lr.coef_):
        print(f"    {feat:<25}: {coef:+.4f}")
    print(f"    {'Intercepto':<25}: {lr.intercept_:+.4f}")

    return lr, metricas

def arbol_decision(X_train, X_test, y_train, y_test):
    titulo("2.2  ÁRBOL DE DECISIÓN", nivel=2)
    dt = DecisionTreeRegressor(max_depth=4, random_state=42)
    dt.fit(X_train, y_train)
    evaluar("Árbol de Decisión — TRAIN", y_train, dt.predict(X_train))
    metricas = evaluar("Árbol de Decisión — TEST",  y_test,  dt.predict(X_test))
    return dt, metricas

def random_forest(X_train, X_test, y_train, y_test):
    titulo("2.3  RANDOM FOREST", nivel=2)
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=5,
        min_samples_split=4, random_state=42
    )
    rf.fit(X_train, y_train)
    evaluar("Random Forest — TRAIN", y_train, rf.predict(X_train))
    metricas = evaluar("Random Forest — TEST",  y_test,  rf.predict(X_test))

    print("\n  Importancia de variables:")
    importancias = sorted(
        zip(FEATURES, rf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    for feat, imp in importancias:
        barra = "█" * int(imp * 40)
        print(f"    {feat:<25}: {imp:.4f}  {barra}")

    return rf, metricas, importancias

#comparar
def comparar_modelos(metricas_lista):
    titulo("2.4  COMPARATIVA DE MODELOS (TEST SET)", nivel=2)
    tabla = pd.DataFrame(metricas_lista)
    print("\n", tabla[["nombre", "r2", "rmse", "mae"]].to_string(index=False))
    mejor = tabla.loc[tabla["r2"].idxmax(), "nombre"]
    print(f"\n  ✔ Mejor modelo (mayor R²): {mejor}")
    return mejor

def predecir_escenarios(lr, dt, rf):
    titulo("2.5  PREDICCIÓN DE ESCENARIOS", nivel=2)

    escenarios = pd.DataFrame([
        {
            "es_hombre": 1, "edad_tramo": 1, "dias_teletrabajo": 0,
            "ingreso_percentil": 3.5, "modo_principal": 3, "proposito_viaje": 0,
            "tiempo_viaje_min": 42, "descripcion": "Hombre, sin teletrabajo, metro",
        },
        {
            "es_hombre": 0, "edad_tramo": 2, "dias_teletrabajo": 5,
            "ingreso_percentil": 2.0, "modo_principal": 1, "proposito_viaje": 1,
            "tiempo_viaje_min": 55, "descripcion": "Mujer, teletrabajo full, caminata",
        },
        {
            "es_hombre": 1, "edad_tramo": 3, "dias_teletrabajo": 2,
            "ingreso_percentil": 4.5, "modo_principal": 0, "proposito_viaje": 0,
            "tiempo_viaje_min": 45, "descripcion": "Hombre mayor, auto, ingreso alto",
        },
        {
            "es_hombre": 0, "edad_tramo": 0, "dias_teletrabajo": 0,
            "ingreso_percentil": 1.5, "modo_principal": 2, "proposito_viaje": 3,
            "tiempo_viaje_min": 52, "descripcion": "Mujer joven, bus, estudia",
        },
    ])

    X_esc  = escenarios[FEATURES].values
    pred_lr = lr.predict(X_esc)
    pred_dt = dt.predict(X_esc)
    pred_rf = rf.predict(X_esc)

    print(f"\n  {'Escenario':<40} {'Reg.Lin':>8} {'Árbol':>7} {'R.Forest':>9}")
    print(f"  {'-'*40} {'-------':>8} {'-----':>7} {'--------':>9}")
    for i, row in escenarios.iterrows():
        print(f"  {row['descripcion']:<40} {pred_lr[i]:>8.3f}"
              f" {pred_dt[i]:>7.3f} {pred_rf[i]:>9.3f}")

    return escenarios, pred_lr, pred_dt, pred_rf


def ejecutar():
    titulo("PARTE 2 — MODELOS PREDICTIVOS DE MOVILIDAD URBANA")

    X_train, X_test, y_train, y_test = preparar_datos()

    lr, met_lr = regresion_lineal(X_train, X_test, y_train, y_test)
    dt, met_dt = arbol_decision(X_train, X_test, y_train, y_test)
    rf, met_rf, importancias = random_forest(X_train, X_test, y_train, y_test)

    mejor = comparar_modelos([met_lr, met_dt, met_rf])
    escenarios, pred_lr, pred_dt, pred_rf = predecir_escenarios(lr, dt, rf)

    return {
        "modelos":     {"lr": lr, "dt": dt, "rf": rf},
        "splits":      (X_train, X_test, y_train, y_test),
        "metricas":    [met_lr, met_dt, met_rf],
        "importancias": importancias,
        "escenarios":  (escenarios, pred_lr, pred_dt, pred_rf),
        "mejor":       mejor,
    }


if __name__ == "__main__":
    ejecutar()

if __name__ == "__main__":
    ejecutar()
