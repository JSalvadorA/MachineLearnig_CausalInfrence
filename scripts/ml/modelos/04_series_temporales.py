"""
04_series_temporales.py - Análisis de Series Temporales

OPTIMIZADO para evitar colapso de sistema:
- Usa scan_parquet() (lazy mode) en lugar de read_parquet()
- Libera memoria explícitamente

Descomposición estacional y análisis de tendencias.
Diseñado principalmente para dataset C (distrito+mes).

Input:  data/ml/C_distritos/dataset.parquet
Output: resultados/C_distritos/series_temporales/

Uso:
    python scripts/ml/modelos/04_series_temporales.py --dataset C
    python scripts/ml/modelos/04_series_temporales.py --dataset C --distrito "SAN ISIDRO"
"""

import sys
import time
import json
import argparse
import gc
from pathlib import Path
from datetime import datetime

import numpy as np
import polars as pl

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Rutas
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent.parent
DATA_DIR = PROJECT_DIR / "data" / "ml"
RESULTS_DIR = PROJECT_DIR / "resultados"


def parse_args():
    parser = argparse.ArgumentParser(description="Análisis de series temporales")
    parser.add_argument("--dataset", "-d", choices=["A", "B", "C"], default="C",
                        help="Dataset (recomendado: C)")
    parser.add_argument("--distrito", type=str, default=None,
                        help="Analizar distrito específico")
    return parser.parse_args()


def analizar_serie(y: np.ndarray, nombre: str):
    """Analiza una serie temporal."""
    results = {"nombre": nombre, "n_puntos": len(y)}

    # Estadísticas básicas
    results["media"] = float(np.mean(y))
    results["std"] = float(np.std(y))
    results["min"] = float(np.min(y))
    results["max"] = float(np.max(y))

    # Test de estacionariedad
    adf = adfuller(y)
    results["adf_statistic"] = float(adf[0])
    results["adf_pvalue"] = float(adf[1])
    results["es_estacionaria"] = adf[1] < 0.05

    print(f"\n--- {nombre} ---")
    print(f"Puntos: {len(y)}")
    print(f"Media: {np.mean(y):,.0f}")
    print(f"Std: {np.std(y):,.0f}")
    print(f"Estacionaria: {'Sí' if results['es_estacionaria'] else 'No'} (p={adf[1]:.4f})")

    # Descomposición si hay suficientes datos
    if len(y) >= 24:
        decomp = seasonal_decompose(y, model='additive', period=12)

        trend = decomp.trend[~np.isnan(decomp.trend)]
        seasonal = decomp.seasonal[~np.isnan(decomp.seasonal)]

        results["tendencia_inicio"] = float(trend[0])
        results["tendencia_fin"] = float(trend[-1])
        results["tendencia_cambio_pct"] = float((trend[-1] / trend[0] - 1) * 100)

        results["estacionalidad_max"] = float(seasonal.max())
        results["estacionalidad_min"] = float(seasonal.min())
        results["estacionalidad_amplitud"] = float(seasonal.max() - seasonal.min())

        print(f"\nTendencia: {trend[0]:,.0f} → {trend[-1]:,.0f} ({results['tendencia_cambio_pct']:+.1f}%)")
        print(f"Estacionalidad: {seasonal.min():+,.0f} a {seasonal.max():+,.0f}")

        # Patrón mensual
        patron_mensual = {}
        for i in range(12):
            vals = seasonal[i::12]
            patron_mensual[i+1] = float(np.nanmean(vals))

        results["patron_mensual"] = patron_mensual

        meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                 "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
        print("\nPatrón mensual:")
        for m, name in enumerate(meses, 1):
            val = patron_mensual[m]
            bar = "+" * int(max(0, val/5000)) if val > 0 else "-" * int(abs(val)/5000)
            print(f"  {name}: {val:>+12,.0f} {bar}")

    return results


def main():
    args = parse_args()

    if args.dataset != "C":
        print("[ADVERTENCIA] Series temporales funciona mejor con dataset C")
        print("              Datasets A y B no tienen estructura temporal directa")

    input_path = DATA_DIR / "C_distritos" / "dataset.parquet"
    output_dir = RESULTS_DIR / "C_distritos" / "series_temporales"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MODELO 4: SERIES TEMPORALES")
    print("=" * 60)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_dir}")

    if not input_path.exists():
        print(f"\n[ERROR] No encontrado: {input_path}")
        print(f"        Ejecuta primero: python scripts/ml/preparar_datos/03_por_distrito.py")
        sys.exit(1)

    # Cargar con lazy mode
    print("\n[1/3] Cargando datos...")
    lf = pl.scan_parquet(input_path)

    # Collect (dataset C es pequeño, ~1872 filas, no requiere límites)
    df = lf.collect()
    print(f"      Filas: {len(df):,}")

    all_results = []

    # Serie agregada (todos los distritos)
    print("\n[2/3] Analizando serie agregada...")

    serie_total = (
        df
        .group_by(["nuanio", "nummes"])
        .agg([
            pl.col("demanda_total_m3").sum().alias("demanda_total"),
            pl.col("mes_absoluto").first(),
        ])
        .sort(["nuanio", "nummes"])
    )

    y_total = serie_total["demanda_total"].to_numpy()
    results_total = analizar_serie(y_total, "TOTAL (todos los distritos)")
    all_results.append(results_total)

    # Distrito específico o top 5
    print("\n[3/3] Analizando por distrito...")

    if args.distrito:
        distritos = [args.distrito]
    else:
        # Top 5 por demanda
        top_distritos = (
            df
            .group_by("nomdis")
            .agg(pl.col("demanda_total_m3").sum())
            .sort("demanda_total_m3", descending=True)
            .head(5)
            ["nomdis"].to_list()
        )
        distritos = top_distritos

    for distrito in distritos:
        df_distrito = df.filter(pl.col("nomdis") == distrito).sort(["nuanio", "nummes"])

        if len(df_distrito) < 12:
            print(f"\n[SKIP] {distrito}: solo {len(df_distrito)} puntos")
            continue

        y = df_distrito["demanda_total_m3"].to_numpy()
        results_dist = analizar_serie(y, distrito)
        all_results.append(results_dist)

    # Guardar resultados
    output = {
        "modelo": "Series Temporales",
        "timestamp": datetime.now().isoformat(),
        "analisis": all_results,
    }

    results_file = output_dir / f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Liberar memoria explícitamente
    del df, lf
    gc.collect()

    print(f"\n[OK] Resultados: {results_file}")


if __name__ == "__main__":
    main()
