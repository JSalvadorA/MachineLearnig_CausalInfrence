"""
01_por_unidad.py - Agregación por Unidad de Uso (codudu)

OPTIMIZADO para evitar colapso de sistema:
- Usa scan_parquet() (lazy mode) en lugar de read_parquet()
- Procesa sin cargar todo en RAM
- Libera memoria explícitamente

Input:  data/consumo_2021_2023.parquet (109M filas, 1.9 GB)
Output: data/ml/A_unidades/dataset.parquet (~3.2M filas)

Uso:
    python scripts/ml/preparar_datos/01_por_unidad.py
    python scripts/ml/preparar_datos/01_por_unidad.py --limit 1000000
"""

import sys
import time
import argparse
import gc
from pathlib import Path

import polars as pl

# Rutas
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent.parent
INPUT_PATH = PROJECT_DIR / "data" / "consumo_2021_2023.parquet"
OUTPUT_DIR = PROJECT_DIR / "data" / "ml" / "A_unidades"
OUTPUT_PATH = OUTPUT_DIR / "dataset.parquet"


def parse_args():
    parser = argparse.ArgumentParser(description="Agregar datos por unidad de uso")
    parser.add_argument("--limit", type=int, default=None, help="Limitar filas (testing)")
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DATASET A: AGREGACIÓN POR UNIDAD DE USO (codudu)")
    print("=" * 60)
    print(f"\nInput:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    if not INPUT_PATH.exists():
        print(f"\n[ERROR] No encontrado: {INPUT_PATH}")
        sys.exit(1)

    start_time = time.time()

    # =====================================================
    # CRÍTICO: Usar scan_parquet (LAZY MODE)
    # NO usar read_parquet() - causa colapso con 109M filas
    # =====================================================
    print("\n[1/3] Escaneando Parquet (LAZY MODE - no carga en RAM)...")
    lf = pl.scan_parquet(INPUT_PATH)

    if args.limit:
        print(f"      Límite: {args.limit:,} filas")
        lf = lf.head(args.limit)

    # Obtener conteo sin cargar datos
    total_rows = lf.select(pl.len()).collect().item()
    print(f"      Filas a procesar: {total_rows:,}")

    # =====================================================
    # Agregación en LAZY MODE
    # Todo se ejecuta de forma optimizada al hacer collect()
    # =====================================================
    print("\n[2/3] Construyendo pipeline de agregación (lazy)...")

    lf_agg = (
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
        .group_by(["codudu", "codcon"])
        .agg([
            pl.col("nomdis").first().alias("distrito"),
            pl.col("coddis").first().alias("cod_distrito"),
            pl.col("nomcat").first().alias("categoria"),
            pl.col("codcat").first().alias("cod_categoria"),
            pl.col("situdu").first().alias("situdu"),
            pl.col("coorpx").first().alias("coord_x"),
            pl.col("coorpy").first().alias("coord_y"),

            pl.col("consumo").mean().alias("consumo_promedio"),
            pl.col("consumo").std().alias("consumo_std"),
            pl.col("consumo").min().alias("consumo_min"),
            pl.col("consumo").max().alias("consumo_max"),
            pl.col("consumo").sum().alias("consumo_total"),
            pl.col("consumo").median().alias("consumo_mediana"),

            (pl.col("importe_agua") + pl.col("importe_alc") + pl.col("cargo_fijo")).sum().alias("importe_total"),
            (pl.col("importe_agua") / pl.col("consumo").replace(0, None)).mean().alias("tarifa_efectiva"),

            pl.col("horas_dia").mean().alias("horas_dia_prom"),
            pl.col("dias_semana").mean().alias("dias_semana_prom"),
            ((pl.col("horas_dia") / 24.0) * (pl.col("dias_semana") / 7.0)).mean().alias("score_calidad"),

            pl.len().alias("registros_totales"),
            pl.col("nuanio").n_unique().alias("anios_activos"),

            pl.col("consumo").filter(pl.col("estacion") == "Verano").mean().alias("consumo_verano"),
            pl.col("consumo").filter(pl.col("estacion") == "Invierno").mean().alias("consumo_invierno"),
        ])
        .with_columns([
            (pl.col("consumo_max") - pl.col("consumo_min")).alias("consumo_rango"),
            (pl.col("consumo_std") / pl.col("consumo_promedio").replace(0, None)).alias("coef_variacion"),
            (pl.col("consumo_verano") - pl.col("consumo_invierno")).alias("delta_estacional"),
            (pl.col("situdu") == "1").cast(pl.Int8).alias("es_subsidiado"),
        ])
    )

    # =====================================================
    # EJECUTAR: collect() ejecuta todo el pipeline optimizado
    # Polars optimiza internamente para minimizar uso de RAM
    # =====================================================
    print("      Ejecutando agregación (esto puede tomar 2-3 min)...")
    df = lf_agg.collect()

    print(f"      Filas resultantes: {len(df):,}")

    # =====================================================
    # GUARDAR con streaming para evitar picos de RAM
    # =====================================================
    print(f"\n[3/3] Guardando...")
    df.write_parquet(OUTPUT_PATH, compression="zstd", compression_level=3)

    # Liberar memoria explícitamente
    del df
    del lf_agg
    del lf
    gc.collect()

    # Resumen
    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Entrada:     {total_rows:>12,} filas")
    print(f"Salida:      Guardado correctamente")
    print(f"Tamaño:      {file_size_mb:>12.1f} MB")
    print(f"Tiempo:      {elapsed:>12.1f} seg")
    print(f"\n[OK] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
