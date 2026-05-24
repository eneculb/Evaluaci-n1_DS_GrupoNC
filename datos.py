# Datos extraídos del Reporte de Movilidad 2025 (DTPM) y dataset sintético para los modelos predictivos

import pandas as pd
import numpy as np

# particion modal 
data_modal = {
    "modo":          ["Bus", "Bus-Metro", "Metro", "Caminata", "Ciclos", "Motos", "Automovil", "Taxi/Apps", "Otros"],
    "pct_viajes":    [12.7,  4.9,         9.4,     30.1,       7.5,      2.4,     28.6,        3.4,         1.0],
    "pct_distancia": [13.4,  10.6,        15.8,    14.8,       3.5,      4.1,     34.1,        1.6,         2.0],
    "dist_prom_km":  [3.96,  8.10,        6.30,    1.83,       1.73,     6.45,    4.47,        1.81,        7.73],
}
df_modal = pd.DataFrame(data_modal)

# tasa de generacion de viajes por genero
data_genero = {
    "genero":   ["Femenino", "Masculino", "Otro"],
    "viajes":   [8192762,    7751250,     256069],
    "personas": [3066801,    2456155,     67850],
    "tasa":     [2.67,       3.16,        3.77],
}
df_genero = pd.DataFrame(data_genero)

# proposito de viaje x genero
data_proposito = {
    "proposito":     ["Compras", "Estudiar", "Llevar personas", "Ocio", "Otro", "Salud", "Trabajar", "Tramites", "Visita cuidado", "Volver hogar"],
    "femenino_pct":  [16,        8,          7,                 4,      4,      6,       19,         3,          1,                32],
    "masculino_pct": [15,        12,         5,                 4,      2,      3,       28,         4,          1,                25],
    "total_pct":     [16,        11,         6,                 4,      3,      5,       23,         4,          1,                28],
}
df_proposito = pd.DataFrame(data_proposito)

# tiempo promedio de viaje x zona
data_tiempos = {
    "zona":          ["Santiago Centro", "Providencia", "Las Condes", "Maipu", "Puente Alto", "La Pintana", "Quilicura", "Lo Barnechea", "San Bernardo", "Pudahuel"],
    "tiempo_min":    [38.2,              40.0,          42.0,         50.0,    55.0,          53.0,         47.0,        61.0,           52.0,           51.0],
    "es_periferia":  [0,                 0,             0,            1,       1,             1,            1,           0,              1,              1],
    "nivel_ingreso": [3,                 4,             5,            2,       2,             1,            2,           5,              2,              2],
}
df_tiempos = pd.DataFrame(data_tiempos)

# viajes en transporte publico x fuente
data_tp = {
    "fuente":    ["Red Movilidad s/evasion", "Red Movilidad", "EOD 2012", "EMS 2024", "EM 2025"],
    "bus":       [1158243,                   1824004,         2365884,    1640852,    2056772],
    "bus_metro": [866346,                    866346,          1159958,    1139391,    794796],
    "metro":     [1656454,                   1656454,         916828,     1414404,    1516212],
}
df_tp = pd.DataFrame(data_tp)
df_tp["total"] = df_tp["bus"] + df_tp["bus_metro"] + df_tp["metro"]

# dataset sintetico para modelos predictivos
# parametros basados en el reporte (N=412, pagina 7)
np.random.seed(42)
N = 412

# genero 181 hombres, 231 mujeres/otro
es_hombre = np.array([1] * 181 + [0] * 231)
np.random.shuffle(es_hombre)

# tramo de edad
# 0=12-20 (46), 1=20-45 (171), 2=45-60 (74), 3=60+ (121)
edad_tramo = np.array([0]*46 + [1]*171 + [2]*74 + [3]*121)
np.random.shuffle(edad_tramo)

# dias de teletrabajo 
dias_teletrabajo = np.random.choice(
    [0, 1, 2, 3, 4, 5], N,
    p=[0.60, 0.08, 0.07, 0.05, 0.04, 0.16]
)

# ingreso percentil 
ingreso_percentil = np.round(np.random.uniform(1, 5, N), 2)

# modo principal 
# 0=auto, 1=caminata, 2=bus, 3=metro, 4=ciclos, 5=otros
modo_principal = np.random.choice(
    [0, 1, 2, 3, 4, 5], N,
    p=[0.286, 0.301, 0.127, 0.094, 0.075, 0.117]
)

# proposito de viaje 
# 0=trabajo, 1=volver_hogar, 2=compras, 3=estudiar, 4=otros
proposito_viaje = np.random.choice(
    [0, 1, 2, 3, 4], N,
    p=[0.23, 0.28, 0.16, 0.11, 0.22]
)

# tiempo de viaje en minutos 
tiempo_base = np.where(modo_principal == 0, 45,
              np.where(modo_principal == 1, 55,
              np.where(modo_principal == 2, 52,
              np.where(modo_principal == 3, 42,
              np.where(modo_principal == 4, 35, 48)))))

tiempo_viaje_min = np.round(
    np.clip(tiempo_base + np.random.normal(0, 8, N), 10, 90), 0
)

# variable objetivo: viajes diarios - tabla 6 (pagina 13)
tasa_base = np.where(es_hombre == 1, 3.16, 2.67)
viajes_diarios = np.round(np.maximum(0,
    tasa_base
    - 0.35 * dias_teletrabajo
    + 0.10 * ingreso_percentil
    + np.random.normal(0, 0.4, N)
), 2)

df_comunas = pd.DataFrame({
    "es_hombre":         es_hombre,
    "edad_tramo":        edad_tramo,
    "dias_teletrabajo":  dias_teletrabajo,
    "ingreso_percentil": ingreso_percentil,
    "modo_principal":    modo_principal,
    "proposito_viaje":   proposito_viaje,
    "tiempo_viaje_min":  tiempo_viaje_min,
    "viajes_diarios":    viajes_diarios
})

FEATURES = [
    "es_hombre", "edad_tramo", "dias_teletrabajo", "ingreso_percentil",
    "modo_principal", "proposito_viaje", "tiempo_viaje_min"
]
TARGET = "viajes_diarios"
