# Datos extraídos del Reporte de Movilidad 2025 (DTPM) y dataset sintético para los modelos predictivos
 
import pandas as pd
import numpy as np
 
# particion modal - em2025
data_modal={
    "modo":          ["Bus", "Bus-Metro", "Metro", "Caminata", "Ciclos", "Motos", "Automóvil", "Taxi/Apps", "Otros"],
    "pct_viajes":    [12.7,   4.9,         9.4,     30.1,       7.5,      2.4,     28.6,        3.4,         1.0],
    "pct_distancia": [13.4,   10.6,        15.8,    14.8,       3.5,      4.1,     34.1,        1.6,         2.0],
    "dist_prom_km":  [3.96,   8.10,        6.30,    1.83,       1.73,     6.45,    4.47,        1.81,        7.73],
}
df_modal = pd.DataFrame(data_modal)
 
# tasa de generacion de viajes por genero
data_genero={
    "genero":   ["Femenino", "Masculino", "Otro"],
    "viajes":   [8_192_762, 7_751_250, 256_069],
    "personas": [3_066_801, 2_456_155,  67_850],
    "tasa":     [2.67, 3.16, 3.77],
}
df_genero = pd.DataFrame(data_genero)
 
# proposito de viaje x genero
data_proposito={
    "proposito":     ["Compras", "Estudiar", "Llevar personas", "Ocio", "Otro", "Salud", "Trabajar", "Trámites", "Visita cuidado", "Volver hogar"],
    "femenino_pct":  [16,         8,          7,                 4,      4,      6,       19,         3,          1,                32],
    "masculino_pct": [15,         12,         5,                 4,      2,      3,       28,         4,          1,                25],
    "total_pct":     [16,         11,         6,                 4,      3,      5,       23,         4,          1,                28],
}
df_proposito = pd.DataFrame(data_proposito)
 
# tiempo promedio de viaje x zona
data_tiempos={
    "zona": ["Santiago Centro", "Providencia", "Las Condes", "Maipú", "Puente Alto", "La Pintana", "Quilicura", "Lo Barnechea", "San Bernardo", "Pudahuel"],
    "tiempo_min":[38.2,          40.0,          42.0,         50.0,    55.0,          53.0,         47.0,        61.0,           52.0,           51.0],
    "es_periferia":[0,           0,             0,            1,       1,             1,            1,           0,              1,              1],         # 1=si ; 2:no
    "nivel_ingreso": [3,         4,             5,            2,       2,             1,            2,           5,              2,              2],         # 1=bajo ; 5=alto
}
df_tiempos = pd.DataFrame(data_tiempos)
 
# viajes en transporte publico x fuente
data_tp={
    "fuente":    ["Red Movilidad s/evasión", "Red Movilidad", "EOD 2012", "EMS 2024", "EM 2025"],
    "bus":       [1_158_243,                  1_824_004,       2_365_884,  1_640_852,  2_056_772],
    "bus_metro": [866_346,                    866_346,         1_159_958,  1_139_391,  794_796],
    "metro":     [1_656_454,                  1_656_454,       916_828,    1_414_404,  1_516_212],
}
df_tp = pd.DataFrame(data_tp)
df_tp["total"]=df_tp["bus"]+df_tp["bus_metro"]+df_tp["metro"]
 
# ddataset sintético para modelos predictivos
# rangos basados en los valores del reporte (2.67 – 3.77)
np.random.seed(42)
N=120
 
df_comunas = pd.DataFrame({
    "cobertura_tp":      np.random.uniform(0.30, 1.00, N),
    "densidad_pob":      np.random.uniform(2_000, 25_000, N),
    "nivel_ingreso":     np.random.uniform(1, 5, N),
    "pct_teletrabajo":   np.random.uniform(0.05, 0.35, N),
    "dist_centro_km":    np.random.uniform(0.5, 30, N),
    "pct_adultos_mayor": np.random.uniform(0.08, 0.30, N),
    "dia_laboral":       np.random.randint(0, 2, N),
})
 
df_comunas["tasa_viajes"]=(
    2.50
    + 0.60    * df_comunas["cobertura_tp"]
    + 0.00003 * df_comunas["densidad_pob"]
    + 0.05    * df_comunas["nivel_ingreso"]
    - 1.20    * df_comunas["pct_teletrabajo"]
    - 0.02    * df_comunas["dist_centro_km"]
    - 0.80    * df_comunas["pct_adultos_mayor"]
    + 0.40    * df_comunas["dia_laboral"]
    + np.random.normal(0, 0.12, N)
)
 
# columnas usadas como predictores
FEATURES=[
    "cobertura_tp", "densidad_pob", "nivel_ingreso", "pct_teletrabajo", "dist_centro_km", "pct_adultos_mayor", "dia_laboral"
]
TARGET = "tasa_viajes"
