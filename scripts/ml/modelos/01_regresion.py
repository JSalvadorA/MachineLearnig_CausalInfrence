"""
01_regresion.py - Predicción de Consumo con XGBoost

OPTIMIZADO para evitar colapso de sistema:
- Usa scan_parquet() (lazy mode) en lugar de read_parquet()
- Límite por defecto de 500k filas para proteger RAM
- Libera memoria explícitamente

Entrena modelo de regresión para predecir consumo promedio.
Funciona con cualquiera de los 3 datasets (A, B, C).

Input:  data/ml/{A_unidades|B_conexiones|C_distritos}/dataset.parquet
Output: resultados/{dataset}/regresion/

Uso:
    python scripts/ml/modelos/01_regresion.py --dataset A
    python scripts/ml/modelos/01_regresion.py --dataset B
    python scripts/ml/modelos/01_regresion.py --dataset C
    python scripts/ml/modelos/01_regresion.py --dataset A --sample 10000
    python scripts/ml/modelos/01_regresion.py --dataset A --no-limit  # Usar todos los datos
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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import xgboost as xgb
except ImportError:
    print("[ERROR] xgboost no instalado. Ejecuta: pip install xgboost")
    sys.exit(1)

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
    parser = argparse.ArgumentParser(description="Regresión XGBoost para predicción de consumo")
    parser.add_argument("--dataset", "-d", choices=["A", "B", "C"], required=True)
    parser.add_argument("--sample", "-s", type=int, default=None, help="Muestra para testing")
    parser.add_argument("--no-limit", action="store_true", help="Usar todos los datos (ignorar límite por defecto)")
    return parser.parse_args()


def main():
    args = parse_args()

    folder, desc = DATASETS[args.dataset]
    input_path = DATA_DIR / folder / "dataset.parquet"
    output_dir = RESULTS_DIR / folder / "regresion"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MODELO 1: REGRESIÓN XGBOOST")
    print("=" * 60)
    print(f"\nDataset: {args.dataset} - {desc}")
    print(f"Input:   {input_path}")
    print(f"Output:  {output_dir}")

    if not input_path.exists():
        print(f"\n[ERROR] No encontrado: {input_path}")
        print(f"        Ejecuta primero: python scripts/ml/preparar_datos/0{ord(args.dataset)-64}_*.py")
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

    # Configurar features según dataset
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

    # Filtrar columnas existentes
    available = df.columns
    features_cat = [f for f in features_cat if f in available]
    features_num = [f for f in features_num if f in available]

    print(f"\nTarget: {target}")
    print(f"Features categóricas: {features_cat}")
    print(f"Features numéricas: {features_num}")

    # Preparar datos
    print("\n[2/4] Preparando datos...")
    df_ml = df.select([target] + features_cat + features_num).drop_nulls()
    pdf = df_ml.to_pandas()

    # Encodear categóricas
    encoders = {}
    for col in features_cat:
        le = LabelEncoder()
        pdf[col] = le.fit_transform(pdf[col].astype(str))
        encoders[col] = le

    X = pdf[features_cat + features_num]
    y = pdf[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"      Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Entrenar
    print("\n[3/4] Entrenando XGBoost...")
    start = time.time()

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    elapsed = time.time() - start
    print(f"      Tiempo: {elapsed:.1f}s")

    # Evaluar
    print("\n[4/4] Evaluando...")
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "-" * 40)
    print("MÉTRICAS")
    print("-" * 40)
    print(f"RMSE:  {rmse:.4f}")
    print(f"MAE:   {mae:.4f}")
    print(f"R²:    {r2:.4f}")

    # Feature importance
    importance = dict(zip(X.columns, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    print("\n" + "-" * 40)
    print("IMPORTANCIA DE FEATURES")
    print("-" * 40)
    for feat, imp in importance.items():
        bar = "█" * int(imp * 40)
        print(f"{feat:<25} {imp:.4f} {bar}")

    # Guardar resultados
    results = {
        "modelo": "XGBoost Regressor",
        "dataset": args.dataset,
        "timestamp": datetime.now().isoformat(),
        "metricas": {"rmse": rmse, "mae": mae, "r2": r2},
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_importance": importance,
        "parametros": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
    }

    results_file = output_dir / f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Liberar memoria explícitamente
    del df, lf, X, y, X_train, X_test, y_train, y_test, model
    gc.collect()

    print(f"\n[OK] Resultados guardados: {results_file}")


if __name__ == "__main__":
    main()
