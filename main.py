import est_desc
import modelos_pred
import visualizacion

def banner(texto):
    linea = "█" * 62
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
    visualizacion.generar_todas(resultados)

    banner("RESUMEN EJECUTIVO")
    mejor   = resultados["mejor"]
    metricas = resultados["metricas"]
    imps    = resultados["importancias"]

    print("""
  PARTE 1 — Hallazgos descriptivos
  ──────────────────────────────────
  • Partición modal: Caminata (30.1%) y Automóvil (28.6%) dominan.
    Bus+Metro recorre la mayor distancia promedio (8.1 km por viaje).
  • Tiempos de viaje: media 49 min. Rango [38–61 min].
    Lo Barnechea (61 min) vs. Santiago Centro (38 min).
  • Tasa de viajes: hombres 3.16 > promedio 3.07 > mujeres 2.67.
  • Teletrabajo correlaciona negativamente con viajes diarios (r = -0.80).
  • Ser hombre correlaciona positivamente con viajes diarios (r = +0.27).
""")

    print("  PARTE 2 — Métricas en test set")
    print("  ──────────────────────────────────")
    for m in metricas:
        print(f"  {m['nombre']:<42} R²={m['r2']:.3f}  RMSE={m['rmse']:.4f}")

    print(f"\n  ✔ Mejor modelo: {mejor}")
    print("\n  Top 3 variables más importantes (Random Forest):")
    for feat, imp in imps[:3]:
        print(f"    • {feat}: {imp:.3f}")

    print("""
  CONCLUSIONES
  ─────────────
  1. El teletrabajo es el factor que más reduce la movilidad diaria.
  2. Los hombres generan más viajes diarios que las mujeres (3.16 vs 2.67).
  3. El ingreso percentil tiene un efecto positivo moderado en los viajes.
  4. El 57% de viajes en TP los realizan mujeres.
  5. La Regresión Lineal obtuvo el mejor R² en este dataset.
""")
