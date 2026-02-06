"""
03_por_distrito.py - Agregación por Distrito y Mes

OPTIMIZADO para evitar colapso de sistema:
- Usa scan_parquet() (lazy mode) en lugar de read_parquet()
- Procesa sin cargar todo en RAM
- Libera memoria explícitamente

Genera dataset ML agregado a nivel geográfico-temporal.
Cada fila representa un distrito en un mes específico.
Ideal para series temporales y análisis de demanda.

Input:  data/consumo_2021_2023.parquet (109M filas)
Output: data/ml/C_distritos/dataset.parquet (~1,872 filas)

Uso:
    python scripts/ml/preparar_datos/03_por_distrito.py
    python scripts/ml/preparar_datos/03_por_distrito.py --limit 1000000
"""

import sys
import time
import argparse
import gc
from pathlib import Path

import polars as pl

# Rutas
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent.parent  # new/
INPUT_PATH = PROJECT_DIR / "data" / "consumo_2021_2023.parquet"
OUTPUT_DIR = PROJECT_DIR / "data" / "ml" / "C_distritos"
OUTPUT_PATH = OUTPUT_DIR / "dataset.parquet"


def parse_args():
    parser = argparse.ArgumentParser(description="Agregar datos por distrito y mes")
    parser.add_argument("--limit", type=int, default=None, help="Limitar filas (testing)")
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DATASET C: AGREGACIÓN POR DISTRITO Y MES")
    print("=" * 60)
    print(f"\nInput:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    if not INPUT_PATH.exists():
        print(f"\n[ERROR] No encontrado: {INPUT_PATH}")
        sys.exit(1)

    start_time = time.time()

    # Leer lazy
    print("\n[1/3] Escaneando Parquet...")
    lf = pl.scan_parquet(INPUT_PATH)

    if args.limit:
        print(f"      Límite: {args.limit:,} filas")
        lf = lf.head(args.limit)

    total_rows = lf.select(pl.len()).collect().item()
    print(f"      Filas: {total_rows:,}")

    # Agregar
    print("\n[2/3] Agregando por distrito + mes...")

    df = (
        lf
        .with_columns([
            pl.col("volfac").cast(pl.Float64).alias("consumo"),
            pl.col("imagua").cast(pl.Float64).alias("importe_agua"),
            pl.col("imalca").cast(pl.Float64).alias("importe_alc"),
            pl.col("imcafi").cast(pl.Float64).alias("cargo_fijo"),
            pl.col("hoxdia").cast(pl.Float64).alias("horas_dia"),
            pl.col("diasem").cast(pl.Float64).alias("dias_semana"),
            ((pl.col("nuanio") - 2021) * 12 + pl.col("nummes")).alias("mes_absoluto"),
            pl.when(pl.col("nummes").is_in([12, 1, 2])).then(pl.lit("Verano"))
              .when(pl.col("nummes").is_in([3, 4, 5])).then(pl.lit("Otono"))
              .when(pl.col("nummes").is_in([6, 7, 8])).then(pl.lit("Invierno"))
              .otherwise(pl.lit("Primavera")).alias("estacion"),
        ])
        .group_by(["coddis", "nomdis", "nuanio", "nummes"])
        .agg([
            pl.col("mes_absoluto").first().alias("mes_absoluto"),
            pl.col("estacion").first().alias("estacion"),

            pl.len().alias("num_registros"),
            pl.col("codudu").n_unique().alias("num_unidades"),
            pl.col("codcon").n_unique().alias("num_conexiones"),

            pl.col("consumo").sum().alias("demanda_total_m3"),
            (pl.col("importe_agua") + pl.col("importe_alc") + pl.col("cargo_fijo")).sum().alias("facturacion_total"),

            pl.col("consumo").mean().alias("consumo_promedio"),
            pl.col("consumo").std().alias("consumo_std"),
            pl.col("consumo").median().alias("consumo_mediana"),
            pl.col("consumo").quantile(0.25).alias("consumo_p25"),
            pl.col("consumo").quantile(0.75).alias("consumo_p75"),
            pl.col("consumo").quantile(0.95).alias("consumo_p95"),

            ((pl.col("horas_dia") / 24.0) * (pl.col("dias_semana") / 7.0)).mean().alias("score_calidad"),

            (pl.col("codcat") == "102").sum().alias("n_domestico"),
            (pl.col("codcat") == "103").sum().alias("n_comercial"),
            (pl.col("codcat") == "104").sum().alias("n_industrial"),
            (pl.col("situdu") == "1").sum().alias("n_subsidiados"),
        ])
        .with_columns([
            (pl.col("n_domestico") / pl.col("num_registros") * 100).alias("pct_domestico"),
            (pl.col("n_comercial") / pl.col("num_registros") * 100).alias("pct_comercial"),
            (pl.col("n_industrial") / pl.col("num_registros") * 100).alias("pct_industrial"),
            (pl.col("n_subsidiados") / pl.col("num_registros") * 100).alias("pct_subsidiado"),
            (pl.col("demanda_total_m3") / pl.col("num_unidades")).alias("consumo_per_unidad"),
            (pl.col("facturacion_total") / pl.col("demanda_total_m3").replace(0, None)).alias("tarifa_promedio"),
            (pl.col("consumo_std") / pl.col("consumo_promedio").replace(0, None)).alias("coef_variacion"),
            (pl.col("consumo_p75") - pl.col("consumo_p25")).alias("iqr_consumo"),
        ])
        .sort(["coddis", "nuanio", "nummes"])
        .collect()
    )

    print(f"      Filas resultantes: {len(df):,}")

    # Guardar métricas ANTES de liberar memoria
    num_rows = len(df)
    n_distritos = df["coddis"].n_unique()
    n_meses = df["mes_absoluto"].n_unique()
    top_distritos = df.group_by("nomdis").agg(pl.col("demanda_total_m3").sum()).sort("demanda_total_m3", descending=True).head(5)

    # Guardar
    print(f"\n[3/3] Guardando...")
    df.write_parquet(OUTPUT_PATH, compression="zstd", compression_level=3)

    # Liberar memoria explícitamente
    del df
    del lf
    gc.collect()

    # Resumen
    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Entrada:     {total_rows:>12,} filas")
    print(f"Salida:      {num_rows:>12,} filas")
    print(f"Reducción:   {total_rows / num_rows:>12.1f}x")
    print(f"Tamaño:      {file_size_mb:>12.1f} MB")
    print(f"Tiempo:      {elapsed:>12.1f} seg")
    print(f"\nDistritos:   {n_distritos}")
    print(f"Meses:       {n_meses}")

    # Top 5 distritos
    print("\nTop 5 distritos por demanda:")
    for row in top_distritos.iter_rows():
        print(f"  {row[0]:<25} {row[1]:>12,.0f} m³")

    print(f"\n[OK] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
