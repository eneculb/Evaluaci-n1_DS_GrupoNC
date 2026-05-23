# parte 1 - estadística dscriptiva
# caclula: media, mediana, moda, varianza, desviación estándar, covarianza, correlación y no me acuerdo qué más.

import numpy as np
import pandas as pd
from datos import (
    df_modal, df_genero, df_tiempos, df_tp, df_comunas, FEATURES
)

# helpers

def titulo(texto, nivel=1):
    if nivel==1:
        linea="="*60
        print(f"\n{linea}\n  {texto}\n{linea}")
    else:
        linea="-"*40
        print(f"\n{linea}\n  {texto}\n{linea}")

# estadisticas completas

def estadisticas_completas(serie: pd.Series, nombre: str) -> dict:
    """
    Calcula el conjunto completo de estadísticas descriptivas:
    media, mediana, moda, varianza (pob. y muestral), desviación estándar, rango, CV, cuartiles, IQR, asimetría, curtosis, error estándar e IC 95%.
    """
    datos=serie.dropna().values
    n=len(datos)

    media=np.mean(datos)
    mediana=np.median(datos)

    # moda redondeando a 2 decimales
    valores, conteos = np.unique(datos.round(2), return_counts=True)
    moda=valores[np.argmax(conteos)]

    var_pob  = np.var(datos, ddof=0)
    var_mues = np.var(datos, ddof=1)
    std_pob  = np.std(datos, ddof=0)
    std_mues = np.std(datos, ddof=1)
    rango    = np.ptp(datos)
    cv       = (std_mues / media)*100 if media!=0 else 0
    q1, q3   = np.percentile(datos, [25, 75])
    iqr      = q3-q1
    asimetria = float(pd.Series(datos).skew())
    curtosis  = float(pd.Series(datos).kurtosis())
    error_std = std_mues/np.sqrt(n)
    ic_low    = media-1.96*error_std
    ic_high   = media+1.96*error_std

    print(f"\n  Variable:{nombre}")
    print(f"    n                    : {n}")
    print(f"    Media                : {media:.4f}")
    print(f"    Mediana              : {mediana:.4f}")
    print(f"    Moda (aprox.)        : {moda:.4f}")
    print(f"    Varianza poblacional : {var_pob:.4f}")
    print(f"    Varianza muestral    : {var_mues:.4f}")
    print(f"    Desv. std pob.       : {std_pob:.4f}")
    print(f"    Desv. std muestral   : {std_mues:.4f}")
    print(f"    Rango                : {rango:.4f}")
    print(f"    Coef. de variación   : {cv:.2f}%")
    print(f"    Q1 / Q3              : {q1:.4f} / {q3:.4f}")
    print(f"    IQR                  : {iqr:.4f}")
    print(f"    Asimetría            : {asimetria:.4f}")
    print(f"    Curtosis             : {curtosis:.4f}")
    print(f"    Error estándar       : {error_std:.4f}")
    print(f"    IC 95%               : [{ic_low:.4f}, {ic_high:.4f}]")

    return {
        "n": n, "media": media, "mediana": mediana, "moda": moda, "var_pob": var_pob, "var_mues": var_mues,
        "std_pob": std_pob, "std_mues": std_mues, "rango": rango, "cv_pct": cv, "q1": q1, "q3": q3, "iqr": iqr,
        "asimetria": asimetria, "curtosis": curtosis, "error_std": error_std, "ic_low": ic_low, "ic_high": ic_high,
    }

# covarianza y correlacion

def covarianza_correlacion(df: pd.DataFrame, col1: str, col2: str) -> tuple:
    """
    Calcula covarianza poblacional, muestral y correlación de Pearson entre dos columnas de un DataFrame.
    """
    x=df[col1].values
    y=df[col2].values
    n=len(x)

    cov_pob  = np.sum((x-x.mean())*(y-y.mean()))/n
    cov_mues = np.sum((x-x.mean())*(y-y.mean()))/(n-1)
    corr     = cov_mues/(np.std(x, ddof=1)*np.std(y, ddof=1))

    print(f"\n  Covarianza [{col1}  vs  {col2}]")
    print(f"    Covarianza poblacional : {cov_pob:.4f}")
    print(f"    Covarianza muestral    : {cov_mues:.4f}")
    print(f"    Correlación de Pearson : {corr:.4f}")

    return cov_mues, corr

def ejecutar():
    titulo("1.1  PARTICIÓN MODAL — Estadísticas Descriptivas")
    estadisticas_completas(df_modal["pct_viajes"],    "% Viajes por modo")
    estadisticas_completas(df_modal["dist_prom_km"],  "Distancia promedio (km)")
    estadisticas_completas(df_modal["pct_distancia"], "% Viajes ponderado por distancia")

    titulo("1.2  TIEMPOS DE VIAJE — Estadísticas Descriptivas")
    estadisticas_completas(df_tiempos["tiempo_min"], "Tiempo de viaje (min)")

    titulo("1.3  TASA DE VIAJES POR GÉNERO")
    estadisticas_completas(df_genero["tasa"], "Tasa viajes/persona")

    titulo("1.4  VIAJES EN TRANSPORTE PÚBLICO")
    estadisticas_completas(df_tp["total"], "Total viajes TP por fuente")

    titulo("1.5  COVARIANZA Y CORRELACIÓN")
    pares = [
        ("cobertura_tp",    "tasa_viajes"),
        ("pct_teletrabajo", "tasa_viajes"),
        ("dist_centro_km",  "tasa_viajes"),
        ("nivel_ingreso",   "tasa_viajes"),
        ("densidad_pob",    "tasa_viajes"),
    ]
    for c1, c2 in pares:
        covarianza_correlacion(df_comunas, c1, c2)

    # matriz de correlación completa
    titulo("1.6  MATRIZ DE CORRELACIÓN COMPLETA", nivel=2)
    cols = FEATURES + ["tasa_viajes"]
    print("\n", df_comunas[cols].corr().round(3).to_string())

if __name__ == "__main__":
    ejecutar()
