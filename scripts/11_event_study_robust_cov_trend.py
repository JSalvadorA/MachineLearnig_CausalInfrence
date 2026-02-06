"""
11_event_study_robust_cov_trend.py - Event Study con covariables y tendencia

Modelo:
  volfac ~ sum_{k != -1} beta_k * I(t_rel = k)
          + covariables + tendencia (mes_abs)

SE robustos cluster por unidad (codcon+codudu).

Input:
  data/causal/event_study_panel.parquet

Outputs:
  resultados/causal/event_study_robust_cov_trend_{tipo}_YYYYMMDD_HHMMSS.json
  informe_latex/outputs/figuras_finales/fig_event_study_{tipo}_cov_trend.png
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    import pandas as pd
except ImportError:
    print("[ERROR] pandas no instalado")
    sys.exit(1)

try:
    import polars as pl
except ImportError:
    print("[ERROR] polars no instalado")
    sys.exit(1)

try:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pyarrow no instalado")
    sys.exit(1)

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent.parent
INPUT_PATH = PROJECT_DIR / "data" / "causal" / "event_study_panel.parquet"
OUTPUT_DIR = PROJECT_DIR / "resultados" / "causal"
FIG_DIR = PROJECT_DIR / "informe_latex" / "outputs" / "figuras_finales"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- Plot style (igual que generar_resumen_estacional.py) ---
PALETTE = {
    "primary": "#1F4E79",
    "primary_light": "#A9C0D9",
    "neutral": "#404040",
    "grid": "#E6E6E6",
}

plt.rcParams.update({
    "figure.dpi": 140,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.edgecolor": PALETTE["neutral"],
    "axes.linewidth": 0.8,
    "xtick.color": PALETTE["neutral"],
    "ytick.color": PALETTE["neutral"],
})


def parse_args():
    parser = argparse.ArgumentParser(description="Event Study robusto con covariables y tendencia")
    parser.add_argument("--input", type=str, default=str(INPUT_PATH))
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--cat-cols", type=str, default="coddis,codcat,codtis,codmof")
    parser.add_argument("--num-cols", type=str, default="hoxdia,diasem")
    return parser.parse_args()


def get_t_vals(input_path, event_type):
    df = (
        pl.scan_parquet(input_path)
        .filter(pl.col("event_type") == event_type)
        .select(pl.col("t_rel").unique())
        .collect()
    )
    t_vals = sorted(df["t_rel"].to_list())
    baseline = -1 if -1 in t_vals else (0 if 0 in t_vals else None)
    if baseline is not None:
        t_vals = [t for t in t_vals if t != baseline]
    return t_vals, baseline


def get_categories(input_path, event_type, cat_cols):
    cats = {}
    for col in cat_cols:
        if not col:
            continue
        vals = (
            pl.scan_parquet(input_path)
            .filter(pl.col("event_type") == event_type)
            .select(pl.col(col).unique())
            .collect()[col]
            .drop_nulls()
            .to_list()
        )
        vals = sorted(list(dict.fromkeys(vals)))
        cats[col] = vals
    return cats


def build_dummy_columns(cat_cols, categories):
    dummy_cols = []
    for col in cat_cols:
        vals = categories.get(col, [])
        if len(vals) <= 1:
            continue
        for v in vals[1:]:
            dummy_cols.append(f"{col}__{v}")
    return dummy_cols


def build_X(pdf, t_vals, cat_cols, num_cols, categories, dummy_cols):
    # Event dummies
    t_rel_arr = pdf["t_rel"].astype(int).to_numpy()
    X = np.ones((len(pdf), 1), dtype=float)
    for t in t_vals:
        X = np.column_stack([X, (t_rel_arr == t).astype(float)])

    # Trend control (calendar time)
    mes_abs = (pdf["nuanio"].astype(int) * 12 + pdf["nummes"].astype(int)).to_numpy()
    X = np.column_stack([X, mes_abs])

    # Numeric covariates
    for c in num_cols:
        if c in pdf.columns:
            X = np.column_stack([X, pdf[c].astype(float).to_numpy()])

    # Categorical covariates
    if cat_cols:
        for col in cat_cols:
            if col in pdf.columns:
                pdf[col] = pd.Categorical(pdf[col], categories=categories.get(col, []))
        dummies = pd.get_dummies(pdf[cat_cols], prefix=cat_cols, prefix_sep="__", drop_first=True)
        dummies = dummies.reindex(columns=dummy_cols, fill_value=0)
        if len(dummies.columns) > 0:
            X = np.column_stack([X, dummies.to_numpy(dtype=float)])

    return X


def ols_cluster_robust(input_path, event_type, t_vals, baseline, cat_cols, num_cols, categories, dummy_cols, batch_size):
    dataset = ds.dataset(str(input_path), format="parquet")
    cols = ["codcon", "codudu", "event_type", "t_rel", "volfac", "nuanio", "nummes"] + cat_cols + num_cols

    # Pass 1
    XtX = None
    Xty = None
    n = 0
    sum_y = 0.0
    sum_y2 = 0.0

    for batch in dataset.to_batches(columns=cols, batch_size=batch_size):
        pdf = batch.to_pandas()
        pdf = pdf[pdf["event_type"] == event_type]
        pdf = pdf.dropna(subset=["t_rel", "volfac", "codcon", "codudu", "nuanio", "nummes"])
        if pdf.empty:
            continue

        X = build_X(pdf, t_vals, cat_cols, num_cols, categories, dummy_cols)
        y = pdf["volfac"].astype(float).to_numpy()

        if XtX is None:
            k = X.shape[1]
            XtX = np.zeros((k, k), dtype=np.float64)
            Xty = np.zeros(k, dtype=np.float64)

        XtX += X.T @ X
        Xty += X.T @ y
        n += len(y)
        sum_y += y.sum()
        sum_y2 += (y * y).sum()

    if n == 0:
        raise SystemExit(f"No hay filas para {event_type}")

    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ Xty

    # Pass 2: SSE + cluster sums
    tmp_dir = OUTPUT_DIR / f"tmp_event_study_cov_trend_{event_type}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_parquet = tmp_dir / "cluster_sums.parquet"

    writer = None
    sse = 0.0

    for batch in dataset.to_batches(columns=cols, batch_size=batch_size):
        pdf = batch.to_pandas()
        pdf = pdf[pdf["event_type"] == event_type]
        pdf = pdf.dropna(subset=["t_rel", "volfac", "codcon", "codudu", "nuanio", "nummes"])
        if pdf.empty:
            continue

        X = build_X(pdf, t_vals, cat_cols, num_cols, categories, dummy_cols)
        y = pdf["volfac"].astype(float).to_numpy()
        y_hat = X @ beta
        resid = y - y_hat
        sse += (resid * resid).sum()

        v = X * resid[:, None]
        v_cols = [f"v{i}" for i in range(X.shape[1])]
        df_v = pd.DataFrame(v, columns=v_cols)
        df_v["codcon"] = pdf["codcon"].astype(str)
        df_v["codudu"] = pdf["codudu"].astype(str)

        grouped = df_v.groupby(["codcon", "codudu"], as_index=False)[v_cols].sum()
        table = pa.Table.from_pandas(grouped, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp_parquet, table.schema)
        writer.write_table(table)

    if writer:
        writer.close()

    sg = (
        pl.scan_parquet(tmp_parquet)
        .group_by(["codcon", "codudu"])
        .sum()
        .collect()
    )
    v_cols = [c for c in sg.columns if c.startswith("v")]
    G = sg.height
    sg_np = sg.select(v_cols).to_numpy()
    meat = sg_np.T @ sg_np

    k = XtX_inv.shape[0]
    df_correction = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 else 1.0
    V = df_correction * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.diag(V))
    t_stats = beta / se

    sst = sum_y2 - (sum_y * sum_y) / n
    r2 = 1.0 - (sse / sst) if sst > 0 else None

    return beta, se, t_stats, n, G, r2


def plot_event_study(res, fig_path):
    t_vals = res["t_vals"]
    coef = res["coef"]
    se = res["std_error"]

    xs = sorted(t_vals)
    ys = [coef[str(t)] for t in xs]
    yerr = [1.96 * se[str(t)] for t in xs]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.errorbar(xs, ys, yerr=yerr, fmt="o-", color=PALETTE["primary"], ecolor=PALETTE["primary_light"], capsize=3)
    ax.axhline(0, color=PALETTE["neutral"], linewidth=0.8)
    ax.axvline(0, color=PALETTE["neutral"], linewidth=0.8, linestyle="--")
    ax.set_title(f"Event Study robusto (cov+trend) - {res['event_type']}")
    ax.set_xlabel("Mes relativo al evento (t_rel)")
    ax.set_ylabel("Coeficiente (delta volfac)")
    ax.grid(axis="y", color=PALETTE["grid"], linestyle="-", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] No encontrado: {input_path}")
        sys.exit(1)

    cat_cols = [c.strip() for c in args.cat_cols.split(",") if c.strip()]
    num_cols = [c.strip() for c in args.num_cols.split(",") if c.strip()]

    for et in ["1to2", "2to1"]:
        t_vals, baseline = get_t_vals(input_path, et)
        if baseline is None:
            print(f"[WARN] No baseline para {et}. Se omite.")
            continue

        categories = get_categories(input_path, et, cat_cols)
        dummy_cols = build_dummy_columns(cat_cols, categories)

        beta, se, t_stats, n, G, r2 = ols_cluster_robust(
            input_path, et, t_vals, baseline, cat_cols, num_cols, categories, dummy_cols, args.batch_size
        )

        # Map coef
        coef = {"intercept": float(beta[0]), "baseline": int(baseline)}
        se_map = {"intercept": float(se[0])}
        t_map = {"intercept": float(t_stats[0])}

        idx = 1
        for t in t_vals:
            coef[str(t)] = float(beta[idx])
            se_map[str(t)] = float(se[idx])
            t_map[str(t)] = float(t_stats[idx])
            idx += 1

        # Trend + covariables indices (not expanded in output)
        res = {
            "event_type": et,
            "baseline": int(baseline),
            "t_vals": t_vals,
            "coef": coef,
            "std_error": se_map,
            "t_stat": t_map,
            "n_obs": int(n),
            "n_clusters": int(G),
            "r2": float(r2) if r2 is not None else None,
            "covariables": {
                "categoricas": cat_cols,
                "numericas": num_cols,
                "tendencia": "mes_abs",
            },
        }

        out_json = OUTPUT_DIR / f"event_study_robust_cov_trend_{et}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

        fig_path = FIG_DIR / f"fig_event_study_{et}_cov_trend.png"
        plot_event_study(res, fig_path)

        print(f"[OK] {out_json}")
        print(f"[OK] {fig_path}")


if __name__ == "__main__":
    main()
