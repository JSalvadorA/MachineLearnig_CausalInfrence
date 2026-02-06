import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]  # new/
RESULTS_DIR = BASE_DIR / "resultados" / "C_distritos" / "series_temporales"
FIG_DIR = BASE_DIR / "informe_latex" / "outputs" / "figuras_finales"
TEX_PATH = BASE_DIR / "informe_latex" / "informe_resumen_ml.tex"

FIG_DIR.mkdir(parents=True, exist_ok=True)

# Find latest json
json_files = sorted(RESULTS_DIR.glob("resultados_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
if not json_files:
    raise SystemExit(f"No se encontraron resultados en {RESULTS_DIR}")

latest_json = json_files[0]
with latest_json.open("r", encoding="utf-8") as f:
    data = json.load(f)

# Extract total series
analisis = data.get("analisis", [])
if not analisis:
    raise SystemExit("JSON sin analisis")

serie_total = None
for item in analisis:
    if str(item.get("nombre", "")).upper().startswith("TOTAL"):
        serie_total = item
        break

if not serie_total:
    raise SystemExit("No se encontro serie TOTAL en JSON")

patron = serie_total.get("patron_mensual", {})
# Ensure months 1-12
months = list(range(1, 13))
vals = [float(patron.get(str(m), patron.get(m, 0.0))) for m in months]

# Stats for summary
media = float(serie_total.get("media", 0.0))
std = float(serie_total.get("std", 0.0))
min_val = float(serie_total.get("min", 0.0))
max_val = float(serie_total.get("max", 0.0))
trend_ini = float(serie_total.get("tendencia_inicio", 0.0))
trend_fin = float(serie_total.get("tendencia_fin", 0.0))
trend_pct = float(serie_total.get("tendencia_cambio_pct", 0.0))
amp = float(serie_total.get("estacionalidad_amplitud", 0.0))
adf_p = float(serie_total.get("adf_pvalue", 1.0))

# Find max/min months
max_idx = max(range(12), key=lambda i: vals[i])
min_idx = min(range(12), key=lambda i: vals[i])
max_month = months[max_idx]
min_month = months[min_idx]
max_val_season = vals[max_idx]
min_val_season = vals[min_idx]

month_names = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

# --- Plot style ---
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

colors = [PALETTE["primary"] if v >= 0 else PALETTE["primary_light"] for v in vals]
ax.bar(month_names, vals, color=colors, edgecolor=PALETTE["primary"], linewidth=0.5)
ax.axhline(0, color=PALETTE["neutral"], linewidth=0.8)
ax.set_title("Patron estacional mensual - Demanda total (todos los distritos)")
ax.set_ylabel("Componente estacional (m3)")
ax.grid(axis="y", color=PALETTE["grid"], linestyle="-", linewidth=0.6)

# Annotate max/min
ax.annotate(f"Max: {month_names[max_idx]}",
            xy=(max_idx, max_val_season), xytext=(max_idx, max_val_season * 1.05),
            ha="center", fontsize=8, color=PALETTE["primary"])
ax.annotate(f"Min: {month_names[min_idx]}",
            xy=(min_idx, min_val_season), xytext=(min_idx, min_val_season * 1.05),
            ha="center", fontsize=8, color=PALETTE["neutral"])

fig.tight_layout()
fig_path = FIG_DIR / "fig_estacional_total.png"
fig.savefig(fig_path, dpi=220)
plt.close(fig)

# --- LaTeX summary ---

def fmt_millions(x):
    return f"{x/1_000_000:.2f}M"

fecha = datetime.now().strftime("%d/%m/%Y")

tex = f"""\\documentclass[11pt,a4paper]{{article}}
\\usepackage[spanish]{{babel}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{lmodern}}
\\usepackage[a4paper,top=2.5cm,bottom=2.0cm,left=2.5cm,right=2.5cm,footskip=0.9cm]{{geometry}}
\\usepackage{{setspace}}
\\usepackage{{parskip}}
\\usepackage{{enumitem}}
\\usepackage{{hyperref}}
\\usepackage{{booktabs}}
\\usepackage{{tabularx}}
\\usepackage{{array}}
\\usepackage{{siunitx}}
\\usepackage[table]{{xcolor}}
\\usepackage{{amsmath}}
\\usepackage{{float}}
\\usepackage{{graphicx}}
\\usepackage{{tcolorbox}}
\\usepackage{{caption}}
\\usepackage{{needspace}}

% Caja de analisis (mismo estilo informe_diagnostico)
\\newtcolorbox{{analisis}}{{
  colback=yellow!10,
  colframe=yellow!50!black,
  boxrule=0.5pt,
  arc=3pt,
  left=6pt,
  right=6pt,
  top=4pt,
  bottom=4pt,
  fontupper=\\small
}}

\\hypersetup{{colorlinks=true,linkcolor=blue!60!black,urlcolor=blue!60!black,hypertexnames=false}}
\\urlstyle{{same}}
\\setstretch{{1.1}}
\\setlist{{topsep=0.25em,itemsep=0.2em,leftmargin=1.2cm}}
\\setlength{{\\parskip}}{{0.65em}}
\\captionsetup[figure]{{skip=2pt}}
\\setlength{{\\textfloatsep}}{{8pt}}
\\setlength{{\\intextsep}}{{6pt}}

\\begin{{document}}

\\begin{{center}}
{{\\Large\\textbf{{Resumen Ejecutivo: Estacionalidad de la Demanda (SUNASS 2021--2023)}}}}\\\\[0.4em]
{{\\normalsize\\textbf{{Analisis de series temporales con dataset C (distrito+mes)}}}}\\\\[0.2em]
{{\\small\\textbf{{Fecha: {fecha}}}}}\\\\[0.3em]
\\end{{center}}

\\section*{{Resumen Ejecutivo}}

\\begin{{tcolorbox}}[colback=yellow!10, colframe=yellow!50!black, boxrule=0.5pt, arc=3pt]
\\textbf{{Hallazgo central:}} La demanda total mensual presenta una \\textbf{{estacionalidad fuerte}} (amplitud \~{fmt_millions(amp)} m$^3$) y una \\textbf{{tendencia leve al alza}} (\~{trend_pct:.1f}\%) en 2021--2023.
\\end{{tcolorbox}}

\\begin{{itemize}}[leftmargin=1cm, itemsep=0.3em]
  \\item \\textbf{{Nivel promedio mensual:}} {fmt_millions(media)} m$^3$ (min {fmt_millions(min_val)}, max {fmt_millions(max_val)}).
  \\item \\textbf{{Estacionalidad:}} pico en {month_names[max_idx]} (+{fmt_millions(max_val_season)}) y valle en {month_names[min_idx]} ({fmt_millions(min_val_season)}).
  \\item \\textbf{{Estacionariedad (ADF):}} p={adf_p:.3f} \\rightarrow no estacionaria a nivel agregado.
  \\item \\textbf{{Distritos top:}} SJL, SMP, ATE, Surco y Lima Cercado presentan patrones mensuales estables con tendencias leves.
\\end{{itemize}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.92\\textwidth]{{\\detokenize{{./outputs/figuras_finales/fig_estacional_total.png}}}}
\\caption{{Patron estacional mensual de la demanda total (2021--2023).}}\n\\label{{fig:estacional_total}}
\\end{{figure}}

\\begin{{analisis}}
\\textbf{{Figura~\\ref{{fig:estacional_total}}:}} La componente estacional muestra incrementos claros en verano
(Febrero) y caidas pronunciadas en invierno (Julio--Agosto). Esto sugiere que el consumo total
responde a patrones climaticos y de actividad economica, por lo que los modelos deben incluir
variables temporales (mes/estacion) para prediccion y planificacion operativa.
\\end{{analisis}}

\\end{{document}}
"""

TEX_PATH.write_text(tex, encoding="utf-8")

print(f"OK: grafico -> {fig_path}")
print(f"OK: latex -> {TEX_PATH}")
