"""
03_por_distrito_duckdb.py - Agregación por Distrito y Mes (DuckDB)

OPTIMIZADO con DuckDB para manejar 109M filas sin colapso:
- Out-of-core real con spill to disk
- PRAGMA memory_limit para control estricto de RAM
- Procesamiento SQL directo sin pasar por Python
- COPY directo a Parquet (sin materializar en RAM)

Input:  data/consumo_2021_2023.parquet (109M filas)
Output: data/ml/C_distritos/dataset.parquet (~1,872 filas)

Requisitos:
    pip install duckdb

Uso:
    python scripts/ml/preparar_datos/duckdb/03_por_distrito_duckdb.py
    python scripts/ml/preparar_datos/duckdb/03_por_distrito_duckdb.py --memory-limit 8GB
"""

import sys
import time
import argparse
from pathlib import Path

try:
    import duckdb
except ImportError:
    print("[ERROR] DuckDB no instalado")
    print("        Ejecuta: pip install duckdb")
    sys.exit(1)

# Rutas
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent.parent.parent
INPUT_PATH = PROJECT_DIR / "data" / "consumo_2021_2023.parquet"
OUTPUT_DIR = PROJECT_DIR / "data" / "ml" / "C_distritos"
OUTPUT_PATH = OUTPUT_DIR / "dataset.parquet"
TEMP_DIR = PROJECT_DIR / "tmp"


def parse_args():
    parser = argparse.ArgumentParser(description="Agregar datos por distrito y mes (DuckDB)")
    parser.add_argument("--memory-limit", type=str, default="10GB",
                        help="Límite de RAM para DuckDB (ej: 8GB, 10GB)")
    parser.add_argument("--temp-dir", type=str, default=None,
                        help="Directorio temporal para spill (default: new/tmp)")
    return parser.parse_args()


def main():
    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Temp directory
    temp_dir = Path(args.temp_dir) if args.temp_dir else TEMP_DIR
    temp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DATASET C: AGREGACIÓN POR DISTRITO Y MES (DuckDB)")
    print("=" * 70)
    print(f"\nInput:        {INPUT_PATH}")
    print(f"Output:       {OUTPUT_PATH}")
    print(f"Memory limit: {args.memory_limit}")
    print(f"Temp dir:     {temp_dir}")

    if not INPUT_PATH.exists():
        print(f"\n[ERROR] No encontrado: {INPUT_PATH}")
        sys.exit(1)

    start_time = time.time()

    # Crear conexión DuckDB
    print("\n[1/3] Configurando DuckDB...")
    con = duckdb.connect()

    # Configurar límites
    con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
    con.execute(f"PRAGMA temp_directory='{temp_dir}'")
    con.execute("PRAGMA threads=4")

    print(f"      Memory limit: {args.memory_limit}")
    print(f"      Temp spill:   {temp_dir}")

    # Verificar tamaño
    print("\n[2/3] Analizando datos...")
    row_count = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{INPUT_PATH}')
    """).fetchone()[0]
    print(f"      Filas a procesar: {row_count:,}")

    # Ejecutar agregación
    print("\n[3/3] Ejecutando agregación y guardando...")
    print("      (esto puede tomar 2-3 min)")

    con.execute(f"""
        COPY (
            WITH base AS (
                SELECT
                    coddis,
                    nomdis,
                    nuanio,
                    nummes,
                    (nuanio - 2021) * 12 + nummes AS mes_absoluto,
                    codudu,
                    codcon,
                    CAST(volfac AS DOUBLE) AS consumo,
                    CAST(imagua AS DOUBLE) AS importe_agua,
                    CAST(imalca AS DOUBLE) AS importe_alc,
                    CAST(imcafi AS DOUBLE) AS cargo_fijo,
                    CAST(hoxdia AS DOUBLE) AS horas_dia,
                    CAST(diasem AS DOUBLE) AS dias_semana,
                    codcat,
                    situdu,
                    CASE
                        WHEN nummes IN (12, 1, 2) THEN 'Verano'
                        WHEN nummes IN (3, 4, 5) THEN 'Otono'
                        WHEN nummes IN (6, 7, 8) THEN 'Invierno'
                        ELSE 'Primavera'
                    END AS estacion
                FROM read_parquet('{INPUT_PATH}')
            )
            SELECT
                coddis,
                nomdis,
                nuanio,
                nummes,
                FIRST(mes_absoluto) AS mes_absoluto,
                FIRST(estacion) AS estacion,

                -- Conteos
                COUNT(*) AS num_registros,
                COUNT(DISTINCT codudu) AS num_unidades,
                COUNT(DISTINCT codcon) AS num_conexiones,

                -- Demanda y facturación
                SUM(consumo) AS demanda_total_m3,
                SUM(importe_agua + importe_alc + cargo_fijo) AS facturacion_total,

                -- Estadísticas de consumo
                AVG(consumo) AS consumo_promedio,
                STDDEV(consumo) AS consumo_std,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY consumo) AS consumo_mediana,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY consumo) AS consumo_p25,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY consumo) AS consumo_p75,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY consumo) AS consumo_p95,

                -- Calidad de servicio
                AVG((horas_dia / 24.0) * (dias_semana / 7.0)) AS score_calidad,

                -- Categorías
                SUM(CASE WHEN codcat = '102' THEN 1 ELSE 0 END) AS n_domestico,
                SUM(CASE WHEN codcat = '103' THEN 1 ELSE 0 END) AS n_comercial,
                SUM(CASE WHEN codcat = '104' THEN 1 ELSE 0 END) AS n_industrial,
                SUM(CASE WHEN situdu = '1' THEN 1 ELSE 0 END) AS n_subsidiados,

                -- Features derivadas
                CAST(SUM(CASE WHEN codcat = '102' THEN 1 ELSE 0 END) AS DOUBLE) /
                COUNT(*) * 100 AS pct_domestico,
                CAST(SUM(CASE WHEN codcat = '103' THEN 1 ELSE 0 END) AS DOUBLE) /
                COUNT(*) * 100 AS pct_comercial,
                CAST(SUM(CASE WHEN codcat = '104' THEN 1 ELSE 0 END) AS DOUBLE) /
                COUNT(*) * 100 AS pct_industrial,
                CAST(SUM(CASE WHEN situdu = '1' THEN 1 ELSE 0 END) AS DOUBLE) /
                COUNT(*) * 100 AS pct_subsidiado,
                SUM(consumo) / COUNT(DISTINCT codudu) AS consumo_per_unidad,
                CASE WHEN SUM(consumo) > 0
                     THEN SUM(importe_agua + importe_alc + cargo_fijo) / SUM(consumo)
                     ELSE NULL END AS tarifa_promedio,
                CASE WHEN AVG(consumo) > 0
                     THEN STDDEV(consumo) / AVG(consumo)
                     ELSE NULL END AS coef_variacion,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY consumo) -
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY consumo) AS iqr_consumo

            FROM base
            GROUP BY coddis, nomdis, nuanio, nummes
            ORDER BY coddis, nuanio, nummes
        ) TO '{OUTPUT_PATH}' (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 3)
    """)

    elapsed = time.time() - start_time

    # Verificar output
    output_rows = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{OUTPUT_PATH}')
    """).fetchone()[0]

    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)

    # Estadísticas
    stats = con.execute(f"""
        SELECT
            COUNT(DISTINCT coddis) as n_distritos,
            COUNT(DISTINCT mes_absoluto) as n_meses
        FROM read_parquet('{OUTPUT_PATH}')
    """).fetchone()

    n_distritos, n_meses = stats

    # Top 5 distritos
    top_dist = con.execute(f"""
        SELECT nomdis, SUM(demanda_total_m3) as total_demanda
        FROM read_parquet('{OUTPUT_PATH}')
        GROUP BY nomdis
        ORDER BY total_demanda DESC
        LIMIT 5
    """).fetchall()

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Entrada:     {row_count:>12,} filas")
    print(f"Salida:      {output_rows:>12,} filas")
    print(f"Reducción:   {row_count / output_rows:>12.1f}x")
    print(f"Tamaño:      {file_size_mb:>12.1f} MB")
    print(f"Tiempo:      {elapsed:>12.1f} seg")
    print(f"\nDistritos:   {n_distritos}")
    print(f"Meses:       {n_meses}")

    print("\nTop 5 distritos por demanda:")
    for distrito, demanda in top_dist:
        print(f"  {distrito:<25} {demanda:>12,.0f} m³")

    print(f"\n[OK] {OUTPUT_PATH}")

    # Cleanup
    con.close()


if __name__ == "__main__":
    main()
