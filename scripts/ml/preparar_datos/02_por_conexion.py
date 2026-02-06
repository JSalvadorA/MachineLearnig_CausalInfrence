"""
02_por_conexion.py - Agregación por Conexión (codcon)

OPTIMIZADO para evitar colapso de sistema:
- Usa scan_parquet() (lazy mode) en lugar de read_parquet()
- Procesa sin cargar todo en RAM
- Libera memoria explícitamente

Genera dataset ML agregado a nivel de conexión/predio.
Cada fila representa una conexión (edificio/casa) con todas sus unidades.

Input:  data/consumo_2021_2023.parquet (109M filas)
Output: data/ml/B_conexiones/dataset.parquet (~1.8M filas)

Uso:
    python scripts/ml/preparar_datos/02_por_conexion.py
    python scripts/ml/preparar_datos/02_por_conexion.py --limit 1000000
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
OUTPUT_DIR = PROJECT_DIR / "data" / "ml" / "B_conexiones"
OUTPUT_PATH = OUTPUT_DIR / "dataset.parquet"


def parse_args():
    parser = argparse.ArgumentParser(description="Agregar datos por conexión")
    parser.add_argument("--limit", type=int, default=None, help="Limitar filas (testing)")
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DATASET B: AGREGACIÓN POR CONEXIÓN (codcon)")
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
    print("\n[2/3] Agregando por codcon...")

    df = (
        lf
        .with_columns([
            pl.col("volfac").cast(pl.Float64).alias("consumo"),
            pl.col("imagua").cast(pl.Float64).alias("importe_agua"),
            pl.col("imalca").cast(pl.Float64).alias("importe_alc"),
            pl.col("imcafi").cast(pl.Float64).alias("cargo_fijo"),
            pl.col("hoxdia").cast(pl.Float64).alias("horas_dia"),
            pl.col("diasem").cast(pl.Float64).alias("dias_semana"),
            pl.when(pl.col("nummes").is_in([12, 1, 2])).then(pl.lit("Verano"))
              .when(pl.col("nummes").is_in([3, 4, 5])).then(pl.lit("Otono"))
              .when(pl.col("nummes").is_in([6, 7, 8])).then(pl.lit("Invierno"))
              .otherwise(pl.lit("Primavera")).alias("estacion"),
        ])
        .group_by("codcon")
        .agg([
            pl.col("nomdis").first().alias("distrito"),
            pl.col("coddis").first().alias("cod_distrito"),
            pl.col("nomcat").first().alias("categoria_principal"),
            pl.col("codcat").first().alias("cod_categoria"),
            pl.col("coorpx").first().alias("coord_x"),
            pl.col("coorpy").first().alias("coord_y"),

            pl.col("codudu").n_unique().alias("num_unidades_uso"),

            pl.col("consumo").mean().alias("consumo_promedio"),
            pl.col("consumo").std().alias("consumo_std"),
            pl.col("consumo").min().alias("consumo_min"),
            pl.col("consumo").max().alias("consumo_max"),
            pl.col("consumo").sum().alias("consumo_total"),

            (pl.col("importe_agua") + pl.col("importe_alc") + pl.col("cargo_fijo")).sum().alias("importe_total"),
            (pl.col("importe_agua") / pl.col("consumo").replace(0, None)).mean().alias("tarifa_efectiva"),

            pl.col("horas_dia").mean().alias("horas_dia_prom"),
            ((pl.col("horas_dia") / 24.0) * (pl.col("dias_semana") / 7.0)).mean().alias("score_calidad"),

            pl.len().alias("registros_totales"),
            pl.col("nuanio").n_unique().alias("anios_activos"),

            (pl.col("situdu") == "1").sum().alias("registros_subsidiados"),

            pl.col("consumo").filter(pl.col("estacion") == "Verano").mean().alias("consumo_verano"),
            pl.col("consumo").filter(pl.col("estacion") == "Invierno").mean().alias("consumo_invierno"),

            pl.col("codcat").n_unique().alias("num_categorias"),
        ])
        .with_columns([
            (pl.col("consumo_max") - pl.col("consumo_min")).alias("consumo_rango"),
            (pl.col("consumo_std") / pl.col("consumo_promedio").replace(0, None)).alias("coef_variacion"),
            (pl.col("consumo_verano") - pl.col("consumo_invierno")).alias("delta_estacional"),
            (pl.col("registros_subsidiados") / pl.col("registros_totales")).alias("pct_subsidiado"),
            (pl.col("consumo_total") / pl.col("num_unidades_uso")).alias("consumo_por_udu"),
            pl.when(pl.col("num_unidades_uso") == 1).then(pl.lit("Unifamiliar"))
              .when(pl.col("num_unidades_uso") <= 4).then(pl.lit("Pequeno"))
              .when(pl.col("num_unidades_uso") <= 10).then(pl.lit("Mediano"))
              .otherwise(pl.lit("Grande")).alias("tipo_predio"),
        ])
        .collect()
    )

    print(f"      Filas resultantes: {len(df):,}")

    # Guardar métricas ANTES de liberar memoria
    num_rows = len(df)
    tipo_predio_dist = df.group_by("tipo_predio").len().sort("len", descending=True)

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

    # Distribución tipo predio
    print("\nTipos de predio:")
    for row in tipo_predio_dist.iter_rows():
        print(f"  {row[0]:<15} {row[1]:>10,}")

    print(f"\n[OK] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
