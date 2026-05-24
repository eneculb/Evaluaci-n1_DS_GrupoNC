# parte 1 - estadistica descriptiva
# calcula: media, mediana, moda, varianza, desviacion estandar, covarianza y correlacion

import numpy as np
import pandas as pd
from datos import df_modal, df_genero, df_tiempos, df_tp, df_comunas, FEATURES

def titulo(texto):
    print(f"\n{'='*50}\n  {texto}\n{'='*50}")

def estadisticas_completas(serie, nombre):
    datos = serie.dropna().values

    media        = np.mean(datos)
    mediana      = np.median(datos)
    varianza     = np.var(datos, ddof=1)
    desv_std     = np.std(datos, ddof=1)

    # moda: valor que mas se repite
    valores, conteos = np.unique(datos.round(2), return_counts=True)
    moda = valores[np.argmax(conteos)]

    print(f"\n  Variable: {nombre}")
    print(f"    Media            : {media:.4f}")
    print(f"    Mediana          : {mediana:.4f}")
    print(f"    Moda             : {moda:.4f}")
    print(f"    Varianza         : {varianza:.4f}")
    print(f"    Desv. estandar   : {desv_std:.4f}")

def covarianza_correlacion(df, col1, col2):
    x = df[col1].values
    y = df[col2].values
    n = len(x)

    cov_pob  = np.sum((x - x.mean()) * (y - y.mean())) / n
    cov_mues = np.sum((x - x.mean()) * (y - y.mean())) / (n - 1)
    corr     = cov_mues / (np.std(x, ddof=1) * np.std(y, ddof=1))

    print(f"\n  [{col1}  vs  {col2}]")
    print(f"    Covarianza poblacional : {cov_pob:.4f}")
    print(f"    Covarianza muestral    : {cov_mues:.4f}")
    print(f"    Correlacion de Pearson : {corr:.4f}")

def ejecutar():
    titulo("1.1  PARTICION MODAL — Estadisticas Descriptivas")
    estadisticas_completas(df_modal["pct_viajes"],    "% Viajes por modo")
    estadisticas_completas(df_modal["dist_prom_km"],  "Distancia promedio (km)")
    estadisticas_completas(df_modal["pct_distancia"], "% Viajes ponderado por distancia")

    titulo("1.2  TIEMPOS DE VIAJE — Estadisticas Descriptivas")
    estadisticas_completas(df_tiempos["tiempo_min"], "Tiempo de viaje (min)")

    titulo("1.3  TASA DE VIAJES POR GENERO")
    estadisticas_completas(df_genero["tasa"], "Tasa viajes/persona")

    titulo("1.4  VIAJES EN TRANSPORTE PUBLICO")
    estadisticas_completas(df_tp["total"], "Total viajes TP por fuente")

    titulo("1.5  COVARIANZA Y CORRELACION")
    pares = [
        ("dias_teletrabajo",  "viajes_diarios"),
        ("ingreso_percentil", "viajes_diarios"),
        ("tiempo_viaje_min",  "viajes_diarios"),
        ("edad_tramo",        "viajes_diarios"),
        ("es_hombre",         "viajes_diarios"),
    ]
    for col1, col2 in pares:
        covarianza_correlacion(df_comunas, col1, col2)

    titulo("1.6  MATRIZ DE CORRELACION COMPLETA")
    cols = FEATURES + ["viajes_diarios"]
    print("\n", df_comunas[cols].corr().round(3).to_string())

if __name__ == "__main__":
    ejecutar()
