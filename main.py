import os
import est_desc
import modelos_pred
import visualizacion

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def banner(texto):
    linea = "█" *62
    print(f"\n{linea}")
    print(f"  {texto}")
    print(linea)

if __name__ == "__main__":
    banner("ANÁLISIS DE MOVILIDAD URBANA — GRAN SANTIAGO")
    print("  Fuente: Reporte de Movilidad 2025 (DTPM, Marzo 2026)\n")

    banner("PARTE 1 — ESTADÍSTICA DESCRIPTIVA")
    est_desc.ejecutar()

    banner("PARTE 2 — MODELOS PREDICTIVOS")
    resultados = modelos_pred.ejecutar()

    banner("GENERANDO FIGURAS")
    visualizacion.generar_todas(resultados, output_dir=OUTPUT_DIR)

    banner("RESUMEN EJECUTIVO")
    mejor   =resultados["mejor"]
    metricas=resultados["metricas"]
    imps    =resultados["importancias"]

    print("""
  PARTE 1 — Hallazgos descriptivos
  ──────────────────────────────────
  • Partición modal: Caminata (30.1%) y Automóvil (28.6%) dominan.
    Bus+Metro recorre la mayor distancia promedio (8.1 km por viaje).
  • Tiempos de viaje: media 49 min. Rango [38–61 min].
    Lo Barnechea (61 min) vs. Santiago Centro (38 min).
  • Tasa de viajes: hombres 3.16 > promedio 3.07 > mujeres 2.67.
  • Teletrabajo correlaciona negativamente con la tasa (r = −0.38).
  • Densidad poblacional es el predictor positivo más fuerte (r = +0.54).
""")
    print("  PARTE 2 — Métricas en test set")
    print("  ──────────────────────────────────")
    for m in metricas:
        print(f"  {m['nombre']:<42} R²={m['r2']:.3f}  RMSE={m['rmse']:.4f}")

    print(f"\n  ✔ Mejor modelo: {mejor}")
    print("\n  Top 3 variables más importantes (Random Forest):")
    for feat, imp in imps[:3]:
        print(f"    • {feat}: {imp:.3f}")

    print(f"""
  CONCLUSIONES
  ─────────────
  1. El teletrabajo reduce la movilidad; 20% lo practica 5+ días/semana.
  2. La cobertura de TP y la densidad son los predictores positivos clave.
  3. Las comunas periféricas tienen tiempos >50 min y menor tasa de viajes.
  4. El 57% de viajes en TP los realizan mujeres.
  5. La Regresión Lineal Múltiple obtuvo el mejor R² en este dataset.

  Figuras disponibles en la carpeta: {OUTPUT_DIR}/
""")
