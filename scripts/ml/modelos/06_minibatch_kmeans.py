"""
06_minibatch_kmeans.py - Segmentation with MiniBatchKMeans

Optimized to avoid crashes:
- Uses scan_parquet() for sample mode
- Supports streaming mode for --no-limit using PyArrow batches
- StandardScaler with partial_fit to avoid loading all data

Input:  data/ml/{A_unidades|B_conexiones|C_distritos}/dataset.parquet
Output: resultados/{dataset}/minibatch_kmeans/

Usage:
    python scripts/ml/modelos/06_minibatch_kmeans.py --dataset A
    python scripts/ml/modelos/06_minibatch_kmeans.py --dataset B
    python scripts/ml/modelos/06_minibatch_kmeans.py --dataset C
    python scripts/ml/modelos/06_minibatch_kmeans.py --dataset A --sample 50000
    python scripts/ml/modelos/06_minibatch_kmeans.py --dataset A --no-limit
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
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

try:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pyarrow not installed. Run: pip install pyarrow")
    sys.exit(1)

# Paths
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
    parser = argparse.ArgumentParser(description="MiniBatchKMeans clustering for segmentation")
    parser.add_argument("--dataset", "-d", choices=["A", "B", "C"], required=True)
    parser.add_argument("--sample", "-s", type=int, default=None, help="Row sample for testing")
    parser.add_argument("--k-min", type=int, default=2, help="Min clusters")
    parser.add_argument("--k-max", type=int, default=8, help="Max clusters")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for streaming")
    parser.add_argument("--sample-for-k", type=int, default=50000, help="Sample size for k selection")
    parser.add_argument("--max-iter", type=int, default=100, help="Max iterations for MiniBatchKMeans")
    parser.add_argument("--no-limit", action="store_true", help="Use all rows (streaming mode)")
    return parser.parse_args()


def get_features(dataset, available_cols):
    if dataset == "A":
        features = ["consumo_promedio", "consumo_std", "tarifa_efectiva",
                    "score_calidad", "registros_totales"]
    elif dataset == "B":
        features = ["consumo_promedio", "num_unidades_uso", "tarifa_efectiva",
                    "score_calidad", "pct_subsidiado"]
    else:  # C
        features = ["consumo_promedio", "demanda_total_m3", "tarifa_promedio",
                    "score_calidad", "num_unidades"]

    return [f for f in features if f in available_cols]


def select_best_k(X_scaled, k_min, k_max):
    scores = {}
    for k in range(k_min, k_max + 1):
        if k >= len(X_scaled):
            break
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores[k] = score
        print(f"  k={k}: silhouette={score:.4f}")

    best_k = max(scores, key=scores.get)
    return best_k, scores


def main():
    args = parse_args()

    folder, desc = DATASETS[args.dataset]
    input_path = DATA_DIR / folder / "dataset.parquet"
    output_dir = RESULTS_DIR / folder / "minibatch_kmeans"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MODELO: MINIBATCH K-MEANS")
    print("=" * 60)
    print(f"\nDataset: {args.dataset} - {desc}")
    print(f"Input:   {input_path}")
    print(f"Output:  {output_dir}")

    if not input_path.exists():
        print(f"\n[ERROR] Not found: {input_path}")
        sys.exit(1)

    if not args.no_limit:
        # Sample mode (in-memory)
        print("\n[1/4] Loading data (sample mode)...")
        lf = pl.scan_parquet(input_path)
        DEFAULT_LIMIT = 100_000

        if args.sample:
            lf = lf.head(args.sample)
            print(f"      Limit: {args.sample:,} rows (user sample)")
        else:
            total_available = lf.select(pl.len()).collect().item()
            if total_available > DEFAULT_LIMIT:
                lf = lf.head(DEFAULT_LIMIT)
                print(f"      Limit: {DEFAULT_LIMIT:,} rows (RAM protection - use --no-limit for all)")
                print("      NOTE: clustering is expensive, use --sample for larger tests")
            else:
                print(f"      Rows: {total_available:,} (all)")

        df = lf.collect()
        print(f"      Rows loaded: {len(df):,}")

        features = get_features(args.dataset, df.columns)
        print(f"\nFeatures: {features}")

        print("\n[2/4] Preparing data...")
        df_ml = df.select(features).drop_nulls()
        X = df_ml.to_pandas()
        print(f"      Valid rows: {len(X):,}")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        print(f"\n[3/4] Selecting best k ({args.k_min}-{args.k_max})...")
        best_k, scores = select_best_k(X_scaled, args.k_min, args.k_max)
        print(f"\n  Best k: {best_k} (silhouette={scores[best_k]:.4f})")

        print(f"\n[4/4] Training final model (k={best_k})...")
        kmeans = MiniBatchKMeans(
            n_clusters=best_k,
            random_state=42,
            n_init=10,
            batch_size=args.batch_size,
            max_iter=args.max_iter,
        )
        labels = kmeans.fit_predict(X_scaled)

        X["cluster"] = labels

        # Cluster profiles
        cluster_profiles = {}
        for cluster_id in range(best_k):
            cluster_data = X[X["cluster"] == cluster_id]
            n = len(cluster_data)
            pct = n / len(X) * 100 if len(X) else 0
            profile = {"n": n, "pct": pct}
            for feat in features:
                profile[feat] = float(cluster_data[feat].mean())
            cluster_profiles[cluster_id] = profile

        # Save results
        results = {
            "modelo": "MiniBatchKMeans",
            "dataset": args.dataset,
            "timestamp": datetime.now().isoformat(),
            "best_k": best_k,
            "silhouette_scores": scores,
            "best_silhouette": scores[best_k],
            "cluster_profiles": cluster_profiles,
            "n_samples": len(X),
            "features": features,
            "mode": "sample",
        }

        results_file = output_dir / f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        assignments_file = output_dir / f"asignaciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        pl.from_pandas(X).write_parquet(assignments_file)

        # Cleanup
        del df, lf, X, X_scaled, kmeans
        gc.collect()

        print(f"\n[OK] Results: {results_file}")
        print(f"[OK] Assignments: {assignments_file}")
        return

    # Streaming mode (no-limit)
    print("\n[1/5] Streaming mode enabled (--no-limit)")
    dataset = ds.dataset(str(input_path), format="parquet")

    # Determine features (from schema)
    available_cols = [f.name for f in dataset.schema]
    features = get_features(args.dataset, available_cols)
    print(f"Features: {features}")

    if not features:
        print("[ERROR] No features found in dataset.")
        sys.exit(1)

    # Pass 1: fit scaler incrementally
    print("\n[2/5] Fitting scaler (streaming)...")
    scaler = StandardScaler()
    total_rows = 0

    scanner = dataset.to_batches(columns=features, batch_size=args.batch_size)
    for batch in scanner:
        pdf = batch.to_pandas()
        pdf = pdf.dropna()
        if pdf.empty:
            continue
        scaler.partial_fit(pdf.values)
        total_rows += len(pdf)

    print(f"      Rows used for scaler: {total_rows:,}")

    # Sample for k selection
    print("\n[3/5] Selecting best k on sample...")
    lf_sample = pl.scan_parquet(input_path).select(features)
    total_available = lf_sample.select(pl.len()).collect().item()
    sample_n = min(args.sample_for_k, total_available)
    df_sample = lf_sample.head(sample_n).collect().drop_nulls()
    if len(df_sample) < args.k_min + 1:
        print("[ERROR] Sample too small for k selection.")
        sys.exit(1)

    X_sample = df_sample.to_pandas().values
    X_sample_scaled = scaler.transform(X_sample)
    best_k, scores = select_best_k(X_sample_scaled, args.k_min, args.k_max)
    print(f"\n  Best k: {best_k} (silhouette={scores[best_k]:.4f})")

    # Pass 2: fit MiniBatchKMeans incrementally
    print("\n[4/5] Training MiniBatchKMeans (streaming)...")
    kmeans = MiniBatchKMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
    )

    scanner = dataset.to_batches(columns=features, batch_size=args.batch_size)
    for batch in scanner:
        pdf = batch.to_pandas()
        pdf = pdf.dropna()
        if pdf.empty:
            continue
        X_batch = scaler.transform(pdf.values)
        kmeans.partial_fit(X_batch)

    # Pass 3: assign clusters + profiles + write parquet
    print("\n[5/5] Writing assignments and profiles...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    assignments_file = output_dir / f"asignaciones_{timestamp}.parquet"

    writer = None
    cluster_counts = np.zeros(best_k, dtype=np.int64)
    cluster_sums = np.zeros((best_k, len(features)), dtype=np.float64)
    n_samples = 0

    scanner = dataset.to_batches(columns=features, batch_size=args.batch_size)
    for batch in scanner:
        pdf = batch.to_pandas()
        pdf = pdf.dropna()
        if pdf.empty:
            continue

        X_batch = scaler.transform(pdf.values)
        labels = kmeans.predict(X_batch)

        # Update profiles using raw values
        for cid in range(best_k):
            mask = labels == cid
            if not mask.any():
                continue
            cluster_counts[cid] += int(mask.sum())
            cluster_sums[cid] += pdf.values[mask].sum(axis=0)

        # Write assignments batch
        pdf_out = pdf.copy()
        pdf_out["cluster"] = labels
        table = pa.Table.from_pandas(pdf_out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(assignments_file, table.schema)
        writer.write_table(table)
        n_samples += len(pdf_out)

    if writer:
        writer.close()

    # Build profiles
    cluster_profiles = {}
    for cid in range(best_k):
        n = int(cluster_counts[cid])
        pct = (n / n_samples * 100) if n_samples else 0
        profile = {"n": n, "pct": pct}
        if n > 0:
            means = cluster_sums[cid] / n
            for i, feat in enumerate(features):
                profile[feat] = float(means[i])
        cluster_profiles[cid] = profile

    results = {
        "modelo": "MiniBatchKMeans",
        "dataset": args.dataset,
        "timestamp": datetime.now().isoformat(),
        "best_k": best_k,
        "silhouette_scores": scores,
        "best_silhouette": scores[best_k],
        "cluster_profiles": cluster_profiles,
        "n_samples": n_samples,
        "features": features,
        "mode": "streaming",
        "batch_size": args.batch_size,
    }

    results_file = output_dir / f"resultados_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # Cleanup
    gc.collect()

    print(f"\n[OK] Results: {results_file}")
    print(f"[OK] Assignments: {assignments_file}")


if __name__ == "__main__":
    main()
