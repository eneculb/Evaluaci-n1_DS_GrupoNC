# visualizacion de datos
# genera figuras del analisis y las guarda como PNG

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datos import df_modal, df_genero, df_tiempos, df_proposito, df_comunas, FEATURES

PALETTE = [
    "#667eea", "#4ecdc4", "#ffd93d", "#ff6b6b", "#a8dadc",
    "#764ba2", "#f093fb", "#43e97b", "#fa709a", "#fee140",
]

# f1
def fig1_particion_modal():
    modos  = df_modal["modo"]
    colors = PALETTE[:len(modos)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("FIGURA 1 — Estadistica Descriptiva: Particion Modal", fontsize=13)

    # 1a. % viajes por modo
    ax = axes[0, 0]
    ax.barh(modos, df_modal["pct_viajes"], color=colors)
    ax.set_xlabel("% de viajes")
    ax.set_title("Distribucion de viajes por modo")
    ax.grid(axis="x")

    # 1b. distancia promedio por modo
    ax = axes[0, 1]
    ax.bar(modos, df_modal["dist_prom_km"], color=colors)
    ax.set_ylabel("km")
    ax.set_title("Distancia promedio de viaje por modo")
    ax.tick_params(axis="x", rotation=40, labelsize=7)
    media_d = df_modal["dist_prom_km"].mean()
    ax.axhline(media_d, color="#ffd93d", ls="--", lw=1.5, label=f"Media = {media_d:.2f} km")
    ax.legend(fontsize=8)
    ax.grid(axis="y")

    # 1c. boxplot % viajes
    ax = axes[1, 0]
    ax.boxplot(df_modal["pct_viajes"], patch_artist=True, vert=False,
               boxprops=dict(facecolor="#667eea", alpha=0.7),
               medianprops=dict(color="#ffd93d", lw=2))
    ax.axvline(df_modal["pct_viajes"].mean(), color="#4ecdc4", ls="--", lw=1.5,
               label=f"Media = {df_modal['pct_viajes'].mean():.1f}%")
    ax.set_title("Boxplot — % de viajes por modo")
    ax.set_xlabel("% de viajes")
    ax.legend(fontsize=8)

    # 1d. viajes vs viajes ponderados por distancia
    ax = axes[1, 1]
    x, w = np.arange(len(modos)), 0.4
    ax.bar(x - w/2, df_modal["pct_viajes"],    w, label="% viajes",    color="#667eea", alpha=0.85)
    ax.bar(x + w/2, df_modal["pct_distancia"], w, label="% distancia", color="#4ecdc4", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(modos, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("%")
    ax.set_title("Viajes vs Viajes ponderados por distancia")
    ax.legend(fontsize=8)
    ax.grid(axis="y")

    plt.tight_layout()
    plt.savefig("fig1_particion_modal.png")
    plt.close()
    print("[OK] Figura 1 guardada: fig1_particion_modal.png")

# f2
def fig2_tiempos_genero():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("FIGURA 2 — Tiempos de Viaje y Tasa por Genero", fontsize=13)

    # 2a. tiempo por zona
    ax = axes[0]
    sorted_df = df_tiempos.sort_values("tiempo_min")
    col_zona = ["#ff6b6b" if p else "#667eea" for p in sorted_df["es_periferia"]]
    ax.barh(sorted_df["zona"], sorted_df["tiempo_min"], color=col_zona)
    ax.axvline(df_tiempos["tiempo_min"].mean(), color="#ffd93d", ls="--", lw=1.5)
    ax.set_xlabel("Minutos")
    ax.set_title("Tiempo promedio de viaje por zona")
    ax.grid(axis="x")

    # 2b. tasa por genero
    ax = axes[1]
    g_colors = ["#f093fb", "#667eea", "#4ecdc4"]
    bars = ax.bar(df_genero["genero"], df_genero["tasa"], color=g_colors, width=0.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03,
                f"{b.get_height():.2f}", ha="center", fontsize=10)
    ax.axhline(3.07, color="#ffd93d", ls="--", lw=1.5, label="Promedio = 3.07")
    ax.set_ylabel("Viajes/persona")
    ax.set_title("Tasa de generacion de viajes por genero")
    ax.set_ylim(0, 4.2)
    ax.legend(fontsize=8)
    ax.grid(axis="y")

    # 2c. top 5 propositos por genero
    ax = axes[2]
    top5 = df_proposito.nlargest(5, "total_pct")
    x, w = np.arange(len(top5)), 0.35
    ax.bar(x - w/2, top5["femenino_pct"],  w, label="Femenino",  color="#f093fb", alpha=0.85)
    ax.bar(x + w/2, top5["masculino_pct"], w, label="Masculino", color="#667eea",  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(top5["proposito"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("%")
    ax.set_title("Top 5 propositos de viaje por genero")
    ax.legend(fontsize=8)
    ax.grid(axis="y")

    plt.tight_layout()
    plt.savefig("fig2_tiempos_genero.png")
    plt.close()
    print("[OK] Figura 2 guardada: fig2_tiempos_genero.png")

# f3
def fig3_correlacion():
    cols        = FEATURES + ["viajes_diarios"]
    matriz_corr = df_comunas[cols].corr()

    plt.figure(figsize=(9, 7))
    plt.title("FIGURA 3 — Matriz de Correlacion de Variables de Movilidad", fontsize=12)
    sns.heatmap(matriz_corr.round(2), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.tight_layout()
    plt.savefig("fig3_correlacion.png")
    plt.close()
    print("[OK] Figura 3 guardada: fig3_correlacion.png")

# f4
def fig4_comparativa_modelos(resultados):
    splits  = resultados["splits"]
    X_train, X_test, y_train, y_test = splits
    modelos = resultados["modelos"]

    info = [
        ("Regresion Lineal",  modelos["lr"].predict(X_test), "#667eea"),
        ("Arbol de Decision", modelos["dt"].predict(X_test), "#4ecdc4"),
        ("Random Forest",     modelos["rf"].predict(X_test), "#ffd93d"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("FIGURA 4 — Comparativa de Modelos Predictivos (Test Set)", fontsize=13)

    for ax, (nombre, y_pred, color) in zip(axes, info):
        # r2 calculado con numpy sin importar sklearn
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2_v   = 1 - ss_res / ss_tot

        ax.scatter(y_test, y_pred, alpha=0.65, color=color, s=40)
        lims = [min(y_test.min(), y_pred.min()) - 0.05,
                max(y_test.max(), y_pred.max()) + 0.05]
        ax.plot(lims, lims, "k--", lw=1.2, alpha=0.7, label="Prediccion perfecta")
        ax.set_xlabel("Tasa real")
        ax.set_ylabel("Tasa predicha")
        ax.set_title(f"{nombre}\nR² = {r2_v:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("fig4_comparativa_modelos.png")
    plt.close()
    print("[OK] Figura 4 guardada: fig4_comparativa_modelos.png")

# f5
def fig5_importancia_escenarios(resultados):
    importancias             = resultados["importancias"]
    escenarios, pred_lr, pred_dt, pred_rf = resultados["escenarios"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("FIGURA 5 — Importancia de Variables y Prediccion de Escenarios", fontsize=13)

    # 5a. importancia random forest
    ax = axes[0]
    feats_sorted = sorted(importancias, key=lambda x: x[1])
    labels_f     = [f.replace("_", " ") for f, _ in feats_sorted]
    vals_f       = [v for _, v in feats_sorted]
    colors_f     = [PALETTE[i % len(PALETTE)] for i in range(len(vals_f))]
    ax.barh(labels_f, vals_f, color=colors_f)
    ax.set_xlabel("Importancia")
    ax.set_title("Importancia de variables (Random Forest)")
    ax.grid(axis="x")

    # 5b. prediccion de escenarios
    ax = axes[1]
    labels_e = escenarios["descripcion"].tolist()
    x_e, w_e = np.arange(len(escenarios)), 0.26
    ax.bar(x_e - w_e, pred_lr, w_e, label="Reg. Lineal",    color="#667eea", alpha=0.85)
    ax.bar(x_e,       pred_dt, w_e, label="Arbol Decision", color="#4ecdc4", alpha=0.85)
    ax.bar(x_e + w_e, pred_rf, w_e, label="Random Forest",  color="#ffd93d", alpha=0.85)
    ax.set_xticks(x_e)
    ax.set_xticklabels(labels_e, rotation=20, ha="right", fontsize=7)
    ax.axhline(3.07, color="#ff6b6b", ls="--", lw=1.2, label="Promedio real")
    ax.set_ylabel("Viajes / persona")
    ax.set_title("Prediccion de escenarios por modelo")
    ax.set_ylim(0, 4.5)
    ax.legend(fontsize=8)
    ax.grid(axis="y")

    plt.tight_layout()
    plt.savefig("fig5_importancia_escenarios.png")
    plt.close()
    print("[OK] Figura 5 guardada: fig5_importancia_escenarios.png")

# f6
def fig6_residuales(resultados):
    splits  = resultados["splits"]
    X_train, X_test, y_train, y_test = splits
    modelos = resultados["modelos"]

    info = [
        ("Regresion Lineal",  modelos["lr"].predict(X_test), "#667eea"),
        ("Arbol de Decision", modelos["dt"].predict(X_test), "#4ecdc4"),
        ("Random Forest",     modelos["rf"].predict(X_test), "#ffd93d"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("FIGURA 6 — Analisis de Residuales por Modelo", fontsize=13)

    for ax, (nombre, y_pred, color) in zip(axes, info):
        residuales = y_test - y_pred
        ax.scatter(y_pred, residuales, alpha=0.6, color=color, s=40)
        ax.axhline(0, color="black", lw=1.5, ls="--")
        sigma = residuales.std()
        ax.axhline( sigma, color="#ffd93d", lw=1, ls=":", label=f"+1σ = {sigma:.3f}")
        ax.axhline(-sigma, color="#ffd93d", lw=1, ls=":")
        ax.set_xlabel("Valores predichos")
        ax.set_ylabel("Residuales")
        ax.set_title(f"Residuales — {nombre}")
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("fig6_residuales.png")
    plt.close()
    print("[OK] Figura 6 guardada: fig6_residuales.png")

# funcion principal
def generar_todas(resultados):
    print("\nGENERANDO FIGURAS...")
    fig1_particion_modal()
    fig2_tiempos_genero()
    fig3_correlacion()
    fig4_comparativa_modelos(resultados)
    fig5_importancia_escenarios(resultados)
    fig6_residuales(resultados)
    print("\nTodas las figuras guardadas.")
