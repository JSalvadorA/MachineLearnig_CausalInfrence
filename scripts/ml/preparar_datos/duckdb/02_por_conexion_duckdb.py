"""
02_por_conexion_duckdb.py - Agregación por Conexión (DuckDB)

OPTIMIZADO con DuckDB para manejar 109M filas sin colapso:
- Out-of-core real con spill to disk
- PRAGMA memory_limit para control estricto de RAM
- Procesamiento SQL directo sin pasar por Python
- COPY directo a Parquet (sin materializar en RAM)

Input:  data/consumo_2021_2023.parquet (109M filas)
Output: data/ml/B_conexiones/dataset.parquet (~1.8M filas)

Requisitos:
    pip install duckdb

Uso:
    python scripts/ml/preparar_datos/duckdb/02_por_conexion_duckdb.py
    python scripts/ml/preparar_datos/duckdb/02_por_conexion_duckdb.py --memory-limit 8GB
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
OUTPUT_DIR = PROJECT_DIR / "data" / "ml" / "B_conexiones"
OUTPUT_PATH = OUTPUT_DIR / "dataset.parquet"
TEMP_DIR = PROJECT_DIR / "tmp"


def parse_args():
    parser = argparse.ArgumentParser(description="Agregar datos por conexión (DuckDB)")
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
    print("DATASET B: AGREGACIÓN POR CONEXIÓN (DuckDB)")
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
    print("      (esto puede tomar 3-5 min)")

    con.execute(f"""
        COPY (
            WITH base AS (
                SELECT
                    codcon,
                    codudu,
                    nomdis,
                    coddis,
                    nomcat,
                    codcat,
                    coorpx,
                    coorpy,
                    situdu,
                    CAST(volfac AS DOUBLE) AS consumo,
                    CAST(imagua AS DOUBLE) AS importe_agua,
                    CAST(imalca AS DOUBLE) AS importe_alc,
                    CAST(imcafi AS DOUBLE) AS cargo_fijo,
                    CAST(hoxdia AS DOUBLE) AS horas_dia,
                    CAST(diasem AS DOUBLE) AS dias_semana,
                    nuanio,
                    nummes,
                    CASE
                        WHEN nummes IN (12, 1, 2) THEN 'Verano'
                        WHEN nummes IN (3, 4, 5) THEN 'Otono'
                        WHEN nummes IN (6, 7, 8) THEN 'Invierno'
                        ELSE 'Primavera'
                    END AS estacion
                FROM read_parquet('{INPUT_PATH}')
            )
            SELECT
                codcon,
                FIRST(nomdis) AS distrito,
                FIRST(coddis) AS cod_distrito,
                FIRST(nomcat) AS categoria_principal,
                FIRST(codcat) AS cod_categoria,
                FIRST(coorpx) AS coord_x,
                FIRST(coorpy) AS coord_y,

                -- Número de unidades
                COUNT(DISTINCT codudu) AS num_unidades_uso,

                -- Métricas de consumo
                AVG(consumo) AS consumo_promedio,
                STDDEV(consumo) AS consumo_std,
                MIN(consumo) AS consumo_min,
                MAX(consumo) AS consumo_max,
                SUM(consumo) AS consumo_total,

                -- Métricas económicas
                SUM(importe_agua + importe_alc + cargo_fijo) AS importe_total,
                AVG(CASE WHEN consumo > 0 THEN importe_agua / consumo ELSE NULL END) AS tarifa_efectiva,

                -- Calidad de servicio
                AVG(horas_dia) AS horas_dia_prom,
                AVG((horas_dia / 24.0) * (dias_semana / 7.0)) AS score_calidad,

                -- Temporales
                COUNT(*) AS registros_totales,
                COUNT(DISTINCT nuanio) AS anios_activos,
                SUM(CASE WHEN situdu = '1' THEN 1 ELSE 0 END) AS registros_subsidiados,

                -- Estacionales
                AVG(CASE WHEN estacion = 'Verano' THEN consumo ELSE NULL END) AS consumo_verano,
                AVG(CASE WHEN estacion = 'Invierno' THEN consumo ELSE NULL END) AS consumo_invierno,

                -- Categorías
                COUNT(DISTINCT codcat) AS num_categorias,

                -- Features derivadas (calculadas después del GROUP BY)
                MAX(consumo) - MIN(consumo) AS consumo_rango,
                CASE WHEN AVG(consumo) > 0
                     THEN STDDEV(consumo) / AVG(consumo)
                     ELSE NULL END AS coef_variacion,
                AVG(CASE WHEN estacion = 'Verano' THEN consumo ELSE NULL END) -
                AVG(CASE WHEN estacion = 'Invierno' THEN consumo ELSE NULL END) AS delta_estacional,
                CAST(SUM(CASE WHEN situdu = '1' THEN 1 ELSE 0 END) AS DOUBLE) /
                COUNT(*) AS pct_subsidiado,
                SUM(consumo) / COUNT(DISTINCT codudu) AS consumo_por_udu,
                CASE
                    WHEN COUNT(DISTINCT codudu) = 1 THEN 'Unifamiliar'
                    WHEN COUNT(DISTINCT codudu) <= 4 THEN 'Pequeno'
                    WHEN COUNT(DISTINCT codudu) <= 10 THEN 'Mediano'
                    ELSE 'Grande'
                END AS tipo_predio

            FROM base
            GROUP BY codcon
        ) TO '{OUTPUT_PATH}' (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 3)
    """)

    elapsed = time.time() - start_time

    # Verificar output
    output_rows = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{OUTPUT_PATH}')
    """).fetchone()[0]

    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)

    # Distribución tipo predio
    tipo_dist = con.execute(f"""
        SELECT tipo_predio, COUNT(*) as n
        FROM read_parquet('{OUTPUT_PATH}')
        GROUP BY tipo_predio
        ORDER BY n DESC
    """).fetchall()

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Entrada:     {row_count:>12,} filas")
    print(f"Salida:      {output_rows:>12,} filas")
    print(f"Reducción:   {row_count / output_rows:>12.1f}x")
    print(f"Tamaño:      {file_size_mb:>12.1f} MB")
    print(f"Tiempo:      {elapsed:>12.1f} seg")

    print("\nTipos de predio:")
    for tipo, count in tipo_dist:
        print(f"  {tipo:<15} {count:>10,}")

    print(f"\n[OK] {OUTPUT_PATH}")

    # Cleanup
    con.close()


if __name__ == "__main__":
    main()
