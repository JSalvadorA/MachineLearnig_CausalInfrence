"""
09_did_1to2_pretrends.py - Chequeo de pre-tendencias (paralelismo)

Usa el panel combinado (tratados + controles) y calcula:
- Promedio de volfac por t_rel (solo pre-tratamiento)
- Tabla de resumen (CSV)
- Grafico con estilo estandar (paleta del informe)

Input:
- data/causal/did_1to2_panel_combined.parquet

Outputs:
- data/causal/did_1to2_pretrends.csv
- informe_latex/outputs/figuras_finales/fig_pretrends_did_1to2.png
"""

from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[3]  # new/
INPUT_PATH = BASE_DIR / "data" / "causal" / "did_1to2_panel_combined.parquet"
CSV_OUT = BASE_DIR / "data" / "causal" / "did_1to2_pretrends.csv"
FIG_DIR = BASE_DIR / "informe_latex" / "outputs" / "figuras_finales"
FIG_OUT = FIG_DIR / "fig_pretrends_did_1to2.png"

FIG_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_PATH.exists():
    raise SystemExit(f"No encontrado: {INPUT_PATH}")

# Cargar y filtrar pre-tratamiento
df = (
    pl.scan_parquet(INPUT_PATH)
    .filter(pl.col("t_rel") < 0)
    .group_by(["t_rel", "treated"])
    .agg(pl.col("volfac").mean().alias("volfac_mean"))
    .collect()
)

# Tabla pivot para CSV
pivot = df.pivot(values="volfac_mean", index="t_rel", on="treated")
pivot = pivot.sort("t_rel")
pivot.write_csv(CSV_OUT)

# Preparar series para grafico
pre = df.sort(["t_rel", "treated"])

treated_series = pre.filter(pl.col("treated") == 1).sort("t_rel")
control_series = pre.filter(pl.col("treated") == 0).sort("t_rel")

t_vals = treated_series["t_rel"].to_list()
t_y = treated_series["volfac_mean"].to_list()
c_vals = control_series["t_rel"].to_list()
c_y = control_series["volfac_mean"].to_list()

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

fig, ax = plt.subplots(figsize=(8.2, 4.2))
ax.plot(t_vals, t_y, marker="o", color=PALETTE["primary"], label="Tratados (1->2)")
ax.plot(c_vals, c_y, marker="o", color=PALETTE["primary_light"], label="Controles (siempre 2)")
ax.axvline(0, color=PALETTE["neutral"], linewidth=0.8, linestyle="--")
ax.set_title("Chequeo de pre-tendencias (t_rel < 0)")
ax.set_xlabel("Meses relativos al evento (t_rel)")
ax.set_ylabel("Volumen facturado promedio (volfac)")
ax.grid(axis="y", color=PALETTE["grid"], linestyle="-", linewidth=0.6)
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig(FIG_OUT, dpi=220)
plt.close(fig)

print(f"OK: tabla -> {CSV_OUT}")
print(f"OK: grafico -> {FIG_OUT}")
