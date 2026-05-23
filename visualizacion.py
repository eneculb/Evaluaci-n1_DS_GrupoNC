#ESTA WEA HAY QUE MODIFICAR BOSAI
# genera figuras del análisis y las guarda como PNG.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import r2_score

from datos import df_modal, df_genero, df_tiempos, df_proposito, df_comunas, FEATURES

# ── Configuración visual global ──────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1626",
    "axes.facecolor":   "#16213e",
    "axes.edgecolor":   "#2a2a3e",
    "axes.labelcolor":  "#c0c8d8",
    "xtick.color":      "#c0c8d8",
    "ytick.color":      "#c0c8d8",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2a3e",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})

PALETTE = [
    "#667eea", "#4ecdc4", "#ffd93d", "#ff6b6b", "#a8dadc",
    "#764ba2", "#f093fb", "#43e97b", "#fa709a", "#fee140",
]
SAVE_KWARGS = dict(dpi=150, bbox_inches="tight", facecolor="#0f1626")

#f1
def fig1_particion_modal(output_dir: str):
    modos =df_modal["modo"]
    colors=PALETTE[:len(modos)]

    fig, axes=plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("FIGURA 1 — Estadística Descriptiva: Partición Modal", fontsize=13, color="#e0e0e0", fontweight="bold", y=1.01)

    # 1a. % viajes por modo
    ax=axes[0, 0]
    bars=ax.barh(modos, df_modal["pct_viajes"], color=colors)
    ax.set_xlabel("% de viajes")
    ax.set_title("Distribución de viajes por modo")
    for b in bars:
        ax.text(b.get_width() + 0.2, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}%", va="center", fontsize=8)
    ax.grid(axis="x")

    # 1b. Distancia promedio por modo
    ax=axes[0, 1]
    ax.bar(modos, df_modal["dist_prom_km"], color=colors)
    ax.set_ylabel("km")
    ax.set_title("Distancia promedio de viaje por modo")
    ax.tick_params(axis="x", rotation=40, labelsize=7)
    media_d=df_modal["dist_prom_km"].mean()
    ax.axhline(media_d, color="#ffd93d", ls="--", lw=1.5, label=f"Media = {media_d:.2f} km")
    ax.legend(fontsize=8)
    ax.grid(axis="y")

    # 1c. Boxplot % viajes
    ax=axes[1, 0]
    ax.boxplot(df_modal["pct_viajes"], patch_artist=True, vert=False, boxprops=dict(facecolor="#667eea", alpha=0.7), medianprops=dict(color="#ffd93d", lw=2))
    ax.axvline(df_modal["pct_viajes"].mean(), color="#4ecdc4", ls="--", lw=1.5, label=f"Media = {df_modal['pct_viajes'].mean():.1f}%")
    ax.set_title("Boxplot — % de viajes por modo")
    ax.set_xlabel("% de viajes")
    ax.legend(fontsize=8)

    # 1d. Viajes vs. viajes ponderados por distancia
    ax=axes[1, 1]
    x, w=np.arange(len(modos)), 0.4
    ax.bar(x-w/2, df_modal["pct_viajes"],    w, label="% viajes",  color="#667eea", alpha=0.85)
    ax.bar(x+w/2, df_modal["pct_distancia"], w, label="% distancia", color="#4ecdc4", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(modos, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("%")
    ax.set_title("Viajes vs. Viajes ponderados por distancia")
    ax.legend(fontsize=8)
    ax.grid(axis="y")

    plt.tight_layout()
    path = f"{output_dir}/fig1_particion_modal.png"
    plt.savefig(path, **SAVE_KWARGS)
    plt.close()
    print(f"[OK] Figura 1 guardada → {path}")

#f2
def fig2_tiempos_genero(output_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("FIGURA 2 — Tiempos de Viaje y Tasa por Género", fontsize=13, color="#e0e0e0", fontweight="bold")

    # 2a. Tiempo por zona
    ax = axes[0]
    sorted_df=df_tiempos.sort_values("tiempo_min")
    col_zona=["#ff6b6b" if p else "#667eea" for p in sorted_df["es_periferia"]]
    ax.barh(sorted_df["zona"], sorted_df["tiempo_min"], color=col_zona)
    ax.axvline(df_tiempos["tiempo_min"].mean(), color="#ffd93d", ls="--", lw=1.5)
    ax.set_xlabel("Minutos")
    ax.set_title("Tiempo promedio de viaje\npor zona")
    leyenda = [Patch(color="#ff6b6b", label="Periferia"), Patch(color="#667eea", label="Centro/Oriente")]
    ax.legend(handles=leyenda, fontsize=8)
    ax.grid(axis="x")

    # 2b. Tasa por género
    ax=axes[1]
    g_colors=["#f093fb", "#667eea", "#4ecdc4"]
    bars2=ax.bar(df_genero["genero"], df_genero["tasa"], color=g_colors, width=0.5)
    for b in bars2:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03,
                f"{b.get_height():.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(3.07, color="#ffd93d", ls="--", lw=1.5, label="Promedio = 3.07")
    ax.set_ylabel("Viajes/persona")
    ax.set_title("Tasa de generación de viajes\npor género")
    ax.set_ylim(0, 4.2)
    ax.legend(fontsize=8)
    ax.grid(axis="y")

    # 2c. Top 5 propósitos por género
    ax=axes[2]
    top5=df_proposito.nlargest(5, "total_pct")
    x, w=np.arange(len(top5)), 0.35
    ax.bar(x-w/2, top5["femenino_pct"],  w, label="Femenino",  color="#f093fb", alpha=0.85)
    ax.bar(x+w/2, top5["masculino_pct"], w, label="Masculino", color="#667eea",  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(top5["proposito"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("%")
    ax.set_title("Top 5 propósitos de viaje\npor género")
    ax.legend(fontsize=8)
    ax.grid(axis="y")

    plt.tight_layout()
    path=f"{output_dir}/fig2_tiempos_genero.png"
    plt.savefig(path, **SAVE_KWARGS)
    plt.close()
    print(f"[OK] Figura 2 guardada → {path}")

# f3
def fig3_correlacion(output_dir: str):
    cols       =FEATURES+["tasa_viajes"]
    matriz_corr=df_comunas[cols].corr()
    labels     =["Cob. TP", "Dens. Pob.", "Ingreso", "Teletrabajo", "Dist. Centro", "Adultos Mayor", "Tasa Viajes"]

    fig, ax=plt.subplots(figsize=(9, 7))
    fig.suptitle("FIGURA 3 — Matriz de Correlación de Variables de Movilidad", fontsize=12, color="#e0e0e0", fontweight="bold")

    im = ax.imshow(matriz_corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matriz_corr.iloc[i, j]
            color_txt = "black" if abs(val) > 0.5 else "#e0e0e0"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color_txt, fontweight="bold")

    plt.tight_layout()
    path = f"{output_dir}/fig3_correlacion.png"
    plt.savefig(path, **SAVE_KWARGS)
    plt.close()
    print(f"[OK] Figura 3 guardada → {path}")

#f4
def fig4_comparativa_modelos(resultados: dict, output_dir: str):
    splits=resultados["splits"]
    _, X_test, _, y_test, _, Xs_test = splits
    modelos=resultados["modelos"]

    info=[
        ("Regresión Lineal",  modelos["lr"].predict(Xs_test), "#667eea"),
        ("Árbol de Decisión", modelos["dt"].predict(X_test),  "#4ecdc4"),
        ("Random Forest",     modelos["rf"].predict(X_test),  "#ffd93d"),
    ]

    fig, axes=plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("FIGURA 4 — Comparativa de Modelos Predictivos (Test Set)", fontsize=13, color="#e0e0e0", fontweight="bold")

    for ax, (nombre, y_pred, color) in zip(axes, info):
        r2_v = r2_score(y_test, y_pred)
        ax.scatter(y_test, y_pred, alpha=0.65, color=color, s=40, edgecolors="none")
        lims=[min(y_test.min(), y_pred.min())-0.05,
                max(y_test.max(), y_pred.max())+0.05]
        ax.plot(lims, lims, "w--", lw=1.2, alpha=0.7, label="Predicción perfecta")
        ax.set_xlabel("Tasa real")
        ax.set_ylabel("Tasa predicha")
        ax.set_title(f"{nombre}\nR² = {r2_v:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    path = f"{output_dir}/fig4_comparativa_modelos.png"
    plt.savefig(path, **SAVE_KWARGS)
    plt.close()
    print(f"[OK] Figura 4 guardada → {path}")

#f5
def fig5_importancia_escenarios(resultados: dict, output_dir: str):
    importancias = resultados["importancias"]
    escenarios, pred_lr, pred_dt, pred_rf = resultados["escenarios"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("FIGURA 5 — Importancia de Variables y Predicción de Escenarios", fontsize=13, color="#e0e0e0", fontweight="bold")

    # 5a. Importancia RF
    ax=axes[0]
    feats_sorted=sorted(importancias, key=lambda x: x[1])
    labels_f=[f.replace("_", " ").title() for f, _ in feats_sorted]
    vals_f  =[v for _, v in feats_sorted]
    colors_f=[PALETTE[i % len(PALETTE)] for i in range(len(vals_f))]
    bars_f  =ax.barh(labels_f, vals_f, color=colors_f)
    ax.set_xlabel("Importancia")
    ax.set_title("Importancia de variables\n(Random Forest)")
    for b in bars_f:
        ax.text(b.get_width() + 0.002, b.get_y() + b.get_height() / 2, f"{b.get_width():.3f}", va="center", fontsize=8)
    ax.grid(axis="x")

    # 5b. Predicción de escenarios
    ax=axes[1]
    labels_e=[d.replace(" ", "\n") for d in escenarios["descripcion"]]
    x_e, w_e=np.arange(len(escenarios)), 0.26
    ax.bar(x_e - w_e, pred_lr, w_e, label="Reg. Lineal",    color="#667eea", alpha=0.85)
    ax.bar(x_e,       pred_dt, w_e, label="Árbol Decisión", color="#4ecdc4", alpha=0.85)
    ax.bar(x_e + w_e, pred_rf, w_e, label="Random Forest",  color="#ffd93d", alpha=0.85)
    ax.set_xticks(x_e)
    ax.set_xticklabels(labels_e, fontsize=7.5)
    ax.axhline(3.07, color="#ff6b6b", ls="--", lw=1.2, label="Promedio real")
    ax.set_ylabel("Tasa viajes / persona")
    ax.set_title("Predicción de escenarios\npor modelo")
    ax.set_ylim(0, 4.0)
    ax.legend(fontsize=8)
    ax.grid(axis="y")

    plt.tight_layout()
    path=f"{output_dir}/fig5_importancia_escenarios.png"
    plt.savefig(path, **SAVE_KWARGS)
    plt.close()
    print(f"[OK] Figura 5 guardada → {path}")


#f6
def fig6_residuales(resultados: dict, output_dir: str):
    splits=resultados["splits"]
    _, X_test, _, y_test, _, Xs_test = splits
    modelos=resultados["modelos"]

    info=[
        ("Regresión Lineal",  modelos["lr"].predict(Xs_test), "#667eea"),
        ("Árbol de Decisión", modelos["dt"].predict(X_test),  "#4ecdc4"),
        ("Random Forest",     modelos["rf"].predict(X_test),  "#ffd93d"),
    ]

    fig, axes=plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("FIGURA 6 — Análisis de Residuales por Modelo", fontsize=13, color="#e0e0e0", fontweight="bold")

    for ax, (nombre, y_pred, color) in zip(axes, info):
        residuales=y_test-y_pred
        ax.scatter(y_pred, residuales, alpha=0.6, color=color, s=40, edgecolors="none")
        ax.axhline(0, color="white", lw=1.5, ls="--")
        sigma = residuales.std()
        ax.axhline( sigma, color="#ffd93d", lw=1, ls=":", label=f"+1σ = {sigma:.3f}")
        ax.axhline(-sigma, color="#ffd93d", lw=1, ls=":")
        ax.set_xlabel("Valores predichos")
        ax.set_ylabel("Residuales")
        ax.set_title(f"Residuales—{nombre}")
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    path=f"{output_dir}/fig6_residuales.png"
    plt.savefig(path, **SAVE_KWARGS)
    plt.close()
    print(f"[OK] Figura 6 guardada → {path}")


#funcion prinvipal

def generar_todas(resultados: dict, output_dir: str = "."):
    print("\n  Geneando figuras...")
    fig1_particion_modal(output_dir)
    fig2_tiempos_genero(output_dir)
    fig3_correlacion(output_dir)
    fig4_comparativa_modelos(resultados, output_dir)
    fig5_importancia_escenarios(resultados, output_dir)
    fig6_residuales(resultados, output_dir)
    print(f"\n Todas las figuras se guardaron  en: {output_dir}/")
