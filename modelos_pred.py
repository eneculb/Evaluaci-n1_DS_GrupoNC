#parte 2— modelos predictivos modalidad urbana
# incluye: Regresión Lineal Múltiple, Árbol de Decisión y Random Forest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

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
def evaluar(nombre: str, y_real, y_pred) -> dict:
    mse =mean_squared_error(y_real, y_pred)
    rmse=np.sqrt(mse)
    mae =mean_absolute_error(y_real, y_pred)
    r2  =r2_score(y_real, y_pred)
    print(f"\n  [{nombre}]")
    print(f"    MSE   : {mse:.4f}")
    print(f"    RMSE  : {rmse:.4f}")
    print(f"    MAE   : {mae:.4f}")
    print(f"    R²    : {r2:.4f}  ({r2*100:.2f}% varianza explicada)")
    return {"nombre": nombre, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

def preparar_datos():
    X=df_comunas[FEATURES].values
    y=df_comunas[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    scaler  =StandardScaler()
    Xs_train=scaler.fit_transform(X_train)
    Xs_test =scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, Xs_train, Xs_test, scaler

#modelos
def regresion_lineal(Xs_train, Xs_test, y_train, y_test):
    titulo("2.1  REGRESIÓN LINEAL MÚLTIPLE", nivel=2)
    lr = LinearRegression()
    lr.fit(Xs_train, y_train)
    evaluar("Regresión Lineal — TRAIN", y_train, lr.predict(Xs_train))
    metricas = evaluar("Regresión Lineal — TEST",  y_test,  lr.predict(Xs_test))

    print("\n  Coeficientes (estandarizados):")
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
def comparar_modelos(metricas_lista: list):
    titulo("2.4  COMPARATIVA DE MODELOS (TEST SET)", nivel=2)
    tabla = pd.DataFrame(metricas_lista)
    print("\n", tabla[["nombre", "r2", "rmse", "mae"]].to_string(index=False))
    mejor = tabla.loc[tabla["r2"].idxmax(), "nombre"]
    print(f"\n  ✔ Mejor modelo (mayor R²): {mejor}")
    return mejor

def predecir_escenarios(lr, dt, rf, scaler):
    titulo("2.5  PREDICCIÓN DE ESCENARIOS", nivel=2)

    escenarios = pd.DataFrame([
        {
            "cobertura_tp": 0.95, "densidad_pob": 22_000, "nivel_ingreso": 3.5, "pct_teletrabajo": 0.10, "dist_centro_km": 2.0, 
            "pct_adultos_mayor": 0.15, "dia_laboral": 1, "descripcion": "Centro bien conectado (laboral)",
        },
        {
            "cobertura_tp": 0.45, "densidad_pob": 5_000, "nivel_ingreso": 1.8, "pct_teletrabajo": 0.08, "dist_centro_km": 25.0,
            "pct_adultos_mayor": 0.20, "dia_laboral": 1, "descripcion": "Periferia alejada (laboral)",
        },
        {
            "cobertura_tp": 0.80, "densidad_pob": 8_000, "nivel_ingreso": 4.8, "pct_teletrabajo": 0.32, "dist_centro_km": 10.0, "pct_adultos_mayor": 0.12, "dia_laboral": 1, "descripcion": "Alto teletrabajo e ingreso",
        },
        {
            "cobertura_tp": 0.70, "densidad_pob": 12_000, "nivel_ingreso": 3.0, "pct_teletrabajo": 0.15, "dist_centro_km": 8.0,
            "pct_adultos_mayor": 0.18, "dia_laboral": 0, "descripcion": "Día no laboral (fin de semana)",
        },
    ])

    X_esc       =escenarios[FEATURES].values
    X_esc_scaled=scaler.transform(X_esc)

    pred_lr=lr.predict(X_esc_scaled)
    pred_dt=dt.predict(X_esc)
    pred_rf=rf.predict(X_esc)

    print(f"\n  {'Escenario':<35} {'Reg.Lin':>8} {'Árbol':>7} {'R.Forest':>9}")
    print(f"  {'-'*35} {'-------':>8} {'-----':>7} {'--------':>9}")
    for i, row in escenarios.iterrows():
        print(f"  {row['descripcion']:<35} {pred_lr[i]:>8.3f}"
              f" {pred_dt[i]:>7.3f} {pred_rf[i]:>9.3f}")

    return escenarios, pred_lr, pred_dt, pred_rf


def ejecutar():
    titulo("PARTE 2 — MODELOS PREDICTIVOS DE MOVILIDAD URBANA")

    X_train, X_test, y_train, y_test, Xs_train, Xs_test, scaler = preparar_datos()

    lr, met_lr=regresion_lineal(Xs_train, Xs_test, y_train, y_test)
    dt, met_dt=arbol_decision(X_train, X_test, y_train, y_test)
    rf, met_rf, importancias=random_forest(X_train, X_test, y_train, y_test)

    mejor=comparar_modelos([met_lr, met_dt, met_rf])
    escenarios, pred_lr, pred_dt, pred_rf=predecir_escenarios(lr, dt, rf, scaler)

    return {
        "modelos": {"lr": lr, "dt": dt, "rf": rf},
        "scaler":  scaler,
        "splits":  (X_train, X_test, y_train, y_test, Xs_train, Xs_test),
        "metricas": [met_lr, met_dt, met_rf],
        "importancias": importancias,
        "escenarios": (escenarios, pred_lr, pred_dt, pred_rf),
        "mejor": mejor,
    }


if __name__ == "__main__":
    ejecutar()
