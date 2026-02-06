"""
03_anomalias.py - Detección de Anomalías con Isolation Forest

OPTIMIZADO para evitar colapso de sistema:
- Usa scan_parquet() (lazy mode) en lugar de read_parquet()
- Límite por defecto de 500k filas para proteger RAM
- Libera memoria explícitamente

Detecta usuarios/conexiones con patrones de consumo anómalos.
Útil para identificar fraudes, fugas, o errores de medición.

Input:  data/ml/{A_unidades|B_conexiones|C_distritos}/dataset.parquet
Output: resultados/{dataset}/anomalias/

Uso:
    python scripts/ml/modelos/03_anomalias.py --dataset A
    python scripts/ml/modelos/03_anomalias.py --dataset B
    python scripts/ml/modelos/03_anomalias.py --dataset A --contamination 0.02
    python scripts/ml/modelos/03_anomalias.py --dataset A --no-limit  # Usar todos los datos
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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Rutas
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent.parent
DATA_DIR = PROJECT_DIR / "data" / "ml"
RESULTS_DIR = PROJECT_DIR / "resultados"

DATASETS = {
    "A": ("A_unidades", "Unidades de Uso", "codudu"),
    "B": ("B_conexiones", "Conexiones", "codcon"),
    "C": ("C_distritos", "Distritos", "nomdis"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Detección de anomalías con Isolation Forest")
    parser.add_argument("--dataset", "-d", choices=["A", "B", "C"], required=True)
    parser.add_argument("--sample", "-s", type=int, default=None, help="Muestra para testing")
    parser.add_argument("--contamination", "-c", type=float, default=0.01, help="% esperado de anomalías")
    parser.add_argument("--no-limit", action="store_true", help="Usar todos los datos (ignorar límite por defecto)")
    return parser.parse_args()


def main():
    args = parse_args()

    folder, desc, id_col = DATASETS[args.dataset]
    input_path = DATA_DIR / folder / "dataset.parquet"
    output_dir = RESULTS_DIR / folder / "anomalias"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MODELO 3: DETECCIÓN DE ANOMALÍAS (ISOLATION FOREST)")
    print("=" * 60)
    print(f"\nDataset: {args.dataset} - {desc}")
    print(f"Input:   {input_path}")
    print(f"Output:  {output_dir}")
    print(f"Contamination: {args.contamination*100:.1f}%")

    if not input_path.exists():
        print(f"\n[ERROR] No encontrado: {input_path}")
        sys.exit(1)

    # Cargar con lazy mode
    print("\n[1/4] Cargando datos...")
    lf = pl.scan_parquet(input_path)

    # Límite por defecto para proteger RAM (500k filas)
    DEFAULT_LIMIT = 500_000

    if args.sample:
        # Usuario especificó muestra explícita
        lf = lf.head(args.sample)
        print(f"      Límite: {args.sample:,} filas (especificado por usuario)")
    elif not args.no_limit:
        # Aplicar límite por defecto
        total_available = lf.select(pl.len()).collect().item()
        if total_available > DEFAULT_LIMIT:
            lf = lf.head(DEFAULT_LIMIT)
            print(f"      Límite: {DEFAULT_LIMIT:,} filas (protección RAM - usa --no-limit para todos)")
        else:
            print(f"      Filas: {total_available:,} (todas)")
    else:
        # Usuario solicitó todos los datos
        total_available = lf.select(pl.len()).collect().item()
        print(f"      Filas: {total_available:,} (sin límite)")

    # Collect después de aplicar filtros
    df = lf.collect()
    print(f"      Filas cargadas: {len(df):,}")

    # Features según dataset
    if args.dataset == "A":
        features = ["consumo_promedio", "consumo_std", "tarifa_efectiva",
                    "score_calidad", "coef_variacion"]
    elif args.dataset == "B":
        features = ["consumo_promedio", "consumo_std", "tarifa_efectiva",
                    "score_calidad", "num_unidades_uso"]
    else:  # C
        features = ["consumo_promedio", "demanda_total_m3", "tarifa_promedio",
                    "coef_variacion", "score_calidad"]

    available = df.columns
    features = [f for f in features if f in available]
    print(f"\nFeatures: {features}")

    # Preparar datos
    print("\n[2/4] Preparando datos...")
    cols_select = features.copy()
    if id_col in available:
        cols_select.append(id_col)

    df_ml = df.select(cols_select).drop_nulls()
    X = df_ml.select(features).to_pandas()
    print(f"      Filas válidas: {len(X):,}")

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entrenar
    print("\n[3/4] Entrenando Isolation Forest...")
    start = time.time()

    model = IsolationForest(
        n_estimators=100,
        contamination=args.contamination,
        random_state=42,
        n_jobs=-1
    )
    predictions = model.fit_predict(X_scaled)

    elapsed = time.time() - start
    print(f"      Tiempo: {elapsed:.1f}s")

    # Resultados
    print("\n[4/4] Analizando resultados...")

    # -1 = anomalía, 1 = normal
    n_anomalias = (predictions == -1).sum()
    pct_anomalias = n_anomalias / len(predictions) * 100

    print(f"\n      Anomalías: {n_anomalias:,} ({pct_anomalias:.2f}%)")

    # Scores
    scores = model.decision_function(X_scaled)

    # Agregar al dataframe
    df_results = df_ml.to_pandas()
    df_results["es_anomalia"] = predictions == -1
    df_results["anomaly_score"] = -scores  # Mayor = más anómalo

    # Top anomalías
    anomalias = df_results[df_results["es_anomalia"]].sort_values("anomaly_score", ascending=False)
    normales = df_results[~df_results["es_anomalia"]]

    print("\n" + "-" * 40)
    print("TOP 10 ANOMALÍAS")
    print("-" * 40)

    for i, (_, row) in enumerate(anomalias.head(10).iterrows()):
        if id_col in row:
            print(f"\n[{i+1}] ID: {row[id_col]}")
        else:
            print(f"\n[{i+1}]")
        print(f"    Score: {row['anomaly_score']:.4f}")
        for feat in features[:3]:
            print(f"    {feat}: {row[feat]:.2f}")

    # Comparación
    print("\n" + "-" * 40)
    print("COMPARACIÓN: ANOMALÍAS VS NORMALES")
    print("-" * 40)
    print(f"{'Feature':<25} {'Normal':>12} {'Anomalía':>12} {'Ratio':>8}")
    print("-" * 60)

    comparison = {}
    for feat in features:
        mean_normal = normales[feat].mean()
        mean_anomalia = anomalias[feat].mean()
        ratio = mean_anomalia / mean_normal if mean_normal != 0 else 0
        comparison[feat] = {"normal": mean_normal, "anomalia": mean_anomalia, "ratio": ratio}
        print(f"{feat:<25} {mean_normal:>12.2f} {mean_anomalia:>12.2f} {ratio:>8.2f}x")

    # Guardar resultados
    results = {
        "modelo": "Isolation Forest",
        "dataset": args.dataset,
        "timestamp": datetime.now().isoformat(),
        "contamination": args.contamination,
        "n_total": len(df_results),
        "n_anomalias": int(n_anomalias),
        "pct_anomalias": pct_anomalias,
        "comparison": comparison,
    }

    results_file = output_dir / f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Guardar anomalías
    anomalias_file = output_dir / f"anomalias_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    pl.from_pandas(anomalias).write_parquet(anomalias_file)

    # Liberar memoria explícitamente
    del df, lf, X, X_scaled, model, df_results
    gc.collect()

    print(f"\n[OK] Resultados: {results_file}")
    print(f"[OK] Anomalías: {anomalias_file}")


if __name__ == "__main__":
    main()
