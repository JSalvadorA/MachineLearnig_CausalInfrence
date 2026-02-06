"""
05_lightgbm.py - Consumo prediction with LightGBM

Optimized to avoid crashes:
- Uses polars scan_parquet() (lazy mode)
- Default row limit to protect RAM
- Explicit memory cleanup

Works with datasets A, B, C.

Input:  data/ml/{A_unidades|B_conexiones|C_distritos}/dataset.parquet
Output: resultados/{dataset}/lightgbm/

Usage:
    python scripts/ml/modelos/05_lightgbm.py --dataset A
    python scripts/ml/modelos/05_lightgbm.py --dataset B
    python scripts/ml/modelos/05_lightgbm.py --dataset C
    python scripts/ml/modelos/05_lightgbm.py --dataset A --sample 10000
    python scripts/ml/modelos/05_lightgbm.py --dataset A --no-limit
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import lightgbm as lgb
except ImportError:
    print("[ERROR] lightgbm not installed. Run: pip install lightgbm")
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
    parser = argparse.ArgumentParser(description="LightGBM regression for consumo prediction")
    parser.add_argument("--dataset", "-d", choices=["A", "B", "C"], required=True)
    parser.add_argument("--sample", "-s", type=int, default=None, help="Row sample for testing")
    parser.add_argument("--no-limit", action="store_true", help="Use all rows (ignore default limit)")
    return parser.parse_args()


def main():
    args = parse_args()

    folder, desc = DATASETS[args.dataset]
    input_path = DATA_DIR / folder / "dataset.parquet"
    output_dir = RESULTS_DIR / folder / "lightgbm"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MODELO: LIGHTGBM REGRESSION")
    print("=" * 60)
    print(f"\nDataset: {args.dataset} - {desc}")
    print(f"Input:   {input_path}")
    print(f"Output:  {output_dir}")

    if not input_path.exists():
        print(f"\n[ERROR] Not found: {input_path}")
        sys.exit(1)

    # Load with lazy mode
    print("\n[1/4] Loading data...")
    lf = pl.scan_parquet(input_path)

    # Default limit to protect RAM
    DEFAULT_LIMIT = 500_000

    if args.sample:
        lf = lf.head(args.sample)
        print(f"      Limit: {args.sample:,} rows (user sample)")
    elif not args.no_limit:
        total_available = lf.select(pl.len()).collect().item()
        if total_available > DEFAULT_LIMIT:
            lf = lf.head(DEFAULT_LIMIT)
            print(f"      Limit: {DEFAULT_LIMIT:,} rows (RAM protection - use --no-limit for all)")
        else:
            print(f"      Rows: {total_available:,} (all)")
    else:
        total_available = lf.select(pl.len()).collect().item()
        print(f"      Rows: {total_available:,} (no limit)")
        if total_available > 2_000_000:
            print("      WARNING: very large dataset may require more RAM")

    # Feature config by dataset
    target = "consumo_promedio"
    if args.dataset == "A":
        features_cat = ["distrito", "categoria", "situdu"]
        features_num = ["horas_dia_prom", "dias_semana_prom", "registros_totales",
                        "consumo_verano", "consumo_invierno", "score_calidad"]
    elif args.dataset == "B":
        features_cat = ["distrito", "categoria_principal", "tipo_predio"]
        features_num = ["num_unidades_uso", "horas_dia_prom", "score_calidad",
                        "pct_subsidiado", "registros_totales"]
    else:  # C
        features_cat = ["nomdis", "estacion"]
        features_num = ["mes_absoluto", "num_unidades", "score_calidad",
                        "pct_domestico", "pct_comercial", "pct_subsidiado"]

    # Select only required columns
    cols_needed = [target] + features_cat + features_num
    cols_available = lf.columns
    cols_final = [c for c in cols_needed if c in cols_available]
    lf = lf.select(cols_final)

    # Collect after filters
    df = lf.collect()
    print(f"      Rows loaded: {len(df):,}")

    # Update feature lists to existing columns
    features_cat = [f for f in features_cat if f in df.columns]
    features_num = [f for f in features_num if f in df.columns]

    print(f"\nTarget: {target}")
    print(f"Categorical features: {features_cat}")
    print(f"Numeric features: {features_num}")

    # Prepare data
    print("\n[2/4] Preparing data...")
    df_ml = df.select([target] + features_cat + features_num).drop_nulls()
    pdf = df_ml.to_pandas()

    # Convert categoricals
    for col in features_cat:
        pdf[col] = pdf[col].astype("category")

    X = pdf[features_cat + features_num]
    y = pdf[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"      Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Train
    print("\n[3/4] Training LightGBM...")
    start = time.time()

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        max_bin=255,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        categorical_feature=features_cat or "auto",
    )

    elapsed = time.time() - start
    print(f"      Time: {elapsed:.1f}s")

    # Evaluate
    print("\n[4/4] Evaluating...")
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "-" * 40)
    print("METRICS")
    print("-" * 40)
    print(f"RMSE:  {rmse:.4f}")
    print(f"MAE:   {mae:.4f}")
    print(f"R2:    {r2:.4f}")

    # Feature importance
    importance = dict(zip(X.columns, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    print("\n" + "-" * 40)
    print("FEATURE IMPORTANCE")
    print("-" * 40)
    for feat, imp in importance.items():
        bar = "#" * int((imp / (max(importance.values()) or 1)) * 40)
        print(f"{feat:<25} {imp:>8.2f} {bar}")

    # Save results
    results = {
        "modelo": "LightGBM Regressor",
        "dataset": args.dataset,
        "timestamp": datetime.now().isoformat(),
        "metricas": {"rmse": rmse, "mae": mae, "r2": r2},
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_importance": importance,
        "parametros": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_bin": 255,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    }

    results_file = output_dir / f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # Cleanup
    del df, lf, X, y, X_train, X_test, y_train, y_test, model
    gc.collect()

    print(f"\n[OK] Results saved: {results_file}")


if __name__ == "__main__":
    main()
