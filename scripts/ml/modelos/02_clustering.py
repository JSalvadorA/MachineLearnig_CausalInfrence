"""
02_clustering.py - Segmentación de Usuarios con K-Means

OPTIMIZADO para evitar colapso de sistema:
- Usa scan_parquet() (lazy mode) en lugar de read_parquet()
- Límite por defecto de 100k filas para proteger RAM (clustering es O(n²))
- Libera memoria explícitamente

Agrupa usuarios/conexiones en segmentos basados en patrones de consumo.
Funciona con cualquiera de los 3 datasets (A, B, C).

Input:  data/ml/{A_unidades|B_conexiones|C_distritos}/dataset.parquet
Output: resultados/{dataset}/clustering/

Uso:
    python scripts/ml/modelos/02_clustering.py --dataset A
    python scripts/ml/modelos/02_clustering.py --dataset B
    python scripts/ml/modelos/02_clustering.py --dataset A --sample 50000
    python scripts/ml/modelos/02_clustering.py --dataset A --no-limit  # Usar todos los datos
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Rutas
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent.parent
DATA_DIR = PROJECT_DIR / "data" / "ml"
RESULTS_DIR = PROJECT_DIR / "resultados"

DATASETS = {
    "A": ("A_unidades", "Unidades de Uso"),
    "B": ("B_conexiones", "Conexiones"),
    "C": ("C_distritos", "Distritos"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Clustering K-Means para segmentación")
    parser.add_argument("--dataset", "-d", choices=["A", "B", "C"], required=True)
    parser.add_argument("--sample", "-s", type=int, default=None, help="Muestra para testing")
    parser.add_argument("--k-min", type=int, default=2, help="Mínimo clusters")
    parser.add_argument("--k-max", type=int, default=8, help="Máximo clusters")
    parser.add_argument("--no-limit", action="store_true", help="Usar todos los datos (ignorar límite por defecto)")
    return parser.parse_args()


def main():
    args = parse_args()

    folder, desc = DATASETS[args.dataset]
    input_path = DATA_DIR / folder / "dataset.parquet"
    output_dir = RESULTS_DIR / folder / "clustering"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MODELO 2: CLUSTERING K-MEANS")
    print("=" * 60)
    print(f"\nDataset: {args.dataset} - {desc}")
    print(f"Input:   {input_path}")
    print(f"Output:  {output_dir}")

    if not input_path.exists():
        print(f"\n[ERROR] No encontrado: {input_path}")
        sys.exit(1)

    # Cargar con lazy mode
    print("\n[1/4] Cargando datos...")
    lf = pl.scan_parquet(input_path)

    # Límite por defecto para proteger RAM (100k filas - clustering es O(n²))
    DEFAULT_LIMIT = 100_000

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
            print(f"      NOTA: Clustering es O(n²), usa --sample para muestras más grandes")
        else:
            print(f"      Filas: {total_available:,} (todas)")
    else:
        # Usuario solicitó todos los datos
        total_available = lf.select(pl.len()).collect().item()
        print(f"      Filas: {total_available:,} (sin límite)")
        if total_available > 500_000:
            print(f"      ADVERTENCIA: Clustering con {total_available:,} filas puede ser lento")

    # Collect después de aplicar filtros
    df = lf.collect()
    print(f"      Filas cargadas: {len(df):,}")

    # Features según dataset
    if args.dataset == "A":
        features = ["consumo_promedio", "consumo_std", "tarifa_efectiva",
                    "score_calidad", "registros_totales"]
    elif args.dataset == "B":
        features = ["consumo_promedio", "num_unidades_uso", "tarifa_efectiva",
                    "score_calidad", "pct_subsidiado"]
    else:  # C
        features = ["consumo_promedio", "demanda_total_m3", "tarifa_promedio",
                    "score_calidad", "num_unidades"]

    available = df.columns
    features = [f for f in features if f in available]
    print(f"\nFeatures: {features}")

    # Preparar datos
    print("\n[2/4] Preparando datos...")
    df_ml = df.select(features).drop_nulls()
    X = df_ml.to_pandas()
    print(f"      Filas válidas: {len(X):,}")

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Buscar k óptimo
    print(f"\n[3/4] Buscando k óptimo ({args.k_min}-{args.k_max})...")
    scores = {}

    for k in range(args.k_min, args.k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores[k] = score
        print(f"  k={k}: silhouette={score:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"\n  Mejor k: {best_k} (silhouette={scores[best_k]:.4f})")

    # Modelo final
    print(f"\n[4/4] Entrenando modelo final (k={best_k})...")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    X["cluster"] = labels

    # Perfiles de clusters
    print("\n" + "-" * 40)
    print("PERFILES DE CLUSTERS")
    print("-" * 40)

    cluster_profiles = {}
    for cluster_id in range(best_k):
        cluster_data = X[X["cluster"] == cluster_id]
        n = len(cluster_data)
        pct = n / len(X) * 100

        profile = {"n": n, "pct": pct}
        print(f"\n[Cluster {cluster_id}] - {n:,} ({pct:.1f}%)")

        for feat in features:
            mean_val = cluster_data[feat].mean()
            profile[feat] = mean_val
            print(f"  {feat:<25} {mean_val:>12.2f}")

        cluster_profiles[cluster_id] = profile

    # Guardar resultados
    results = {
        "modelo": "K-Means Clustering",
        "dataset": args.dataset,
        "timestamp": datetime.now().isoformat(),
        "best_k": best_k,
        "silhouette_scores": scores,
        "best_silhouette": scores[best_k],
        "cluster_profiles": cluster_profiles,
        "n_samples": len(X),
        "features": features,
    }

    results_file = output_dir / f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Guardar asignaciones
    assignments_file = output_dir / f"asignaciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    pl.from_pandas(X).write_parquet(assignments_file)

    # Liberar memoria explícitamente
    del df, lf, X, X_scaled, kmeans
    gc.collect()

    print(f"\n[OK] Resultados: {results_file}")
    print(f"[OK] Asignaciones: {assignments_file}")


if __name__ == "__main__":
    main()
