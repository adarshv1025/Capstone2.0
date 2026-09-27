"""
dashboard.py

CLAUDE.md 4.3: the ESG Dashboard, the literal 9th layer of the original
9-layer framework (section 1) that hadn't been visually built until now.

Renders ONE self-contained HTML file (Chart.js via CDN, no build step, no
server) from:
  - a fresh run of the multi-agent pipeline (for the summary cards and the
    per-route planned-vs-actual chart), and
  - the evaluation harness's comparison results (for the multi-agent vs.
    baselines chart) -- prefers evaluation_results/summary_results.csv
    (the multi-snapshot mean, CLAUDE.md 4.2 item 1) if present, falling
    back to a fresh single-snapshot run_evaluation() otherwise.

Chart/color design follows the dataviz skill's method: categorical hues
assigned in fixed slot order (never cycled), status colors (green/red)
reserved for feasibility state and never reused as a categorical series,
legends present for every multi-series chart, stat tiles for headline
numbers. Palette values are the skill's validated default instance.

No lat/lon data is available anywhere in this project (checked: neither
sample_costs/*.csv nor any gnn/ file carries node coordinates), so the
spec's optional Leaflet.js route map is skipped -- would need that data
plumbed through from the graph-construction layer, which is a teammate's
layer, not built here.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gnn"))

from evaluation_harness import run_evaluation
from golden_scenarios import mixed_priority_scenario
from pipeline import run_full_pipeline

# Palette (dataviz skill's validated default instance, references/palette.md).
CATEGORICAL = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
}
STATUS = {"good": "#0ca30c", "critical": "#d03b3b"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781"}
SURFACE = {"chart": "#fcfcfb", "page": "#f9f9f7", "grid": "#e1e0d9", "border": "rgba(11,11,11,0.10)"}

SYSTEM_LABELS = {
    "multi_agent": "Multi-Agent System",
    "baseline_time_only": "Baseline (time-only)",
    "baseline_carbon_only": "Baseline (carbon-only)",
}
SYSTEM_COLORS = {
    "multi_agent": CATEGORICAL["blue"],
    "baseline_time_only": CATEGORICAL["orange"],
    "baseline_carbon_only": CATEGORICAL["aqua"],
}


def _fmt(value, unit="", decimals=1):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    return f"{value:,.{decimals}f}{unit}"


def build_dashboard_html(costs_df: pd.DataFrame, comparison_df: pd.DataFrame, model: str = None, seed: int = None) -> str:
    """
    Runs the mixed_priority golden scenario through the full pipeline for
    the per-route detail, and uses `comparison_df` (one row per
    (scenario, system), from either run_evaluation() or the multi-snapshot
    summary) for the systems-comparison chart. Returns the complete HTML
    document as a string.
    """
    requests, vehicles = mixed_priority_scenario(costs_df)
    result = run_full_pipeline(requests, vehicles, costs_df, model=model, seed=seed)

    metrics = result["esg_report"]["metrics"]
    summary_text = result["esg_report"]["summary"] or "(narrative unavailable)"

    reports_by_id = {rep["request_id"]: rep for rep in result["execution_reports"]}
    per_route = [
        {
            "request_id": rid,
            "planned_time_s": rep["planned_time_s"],
            "actual_time_s": rep["actual_time_s"],
        }
        for rid, rep in reports_by_id.items()
    ]

    # Prefer mean columns (multi-snapshot summary) if present, else the
    # single-run columns run_evaluation() produces directly.
    has_mean = "total_planned_time_s_mean" in comparison_df.columns
    time_col = "total_planned_time_s_mean" if has_mean else "total_planned_time_s"
    carbon_col = "total_planned_carbon_kg_mean" if has_mean else "total_planned_carbon_kg"

    scenarios = sorted(comparison_df["scenario"].unique())
    systems = [s for s in ("multi_agent", "baseline_time_only", "baseline_carbon_only") if s in comparison_df["system"].unique()]
    comparison_chart_data = {
        "scenarios": scenarios,
        "datasets_time": [
            {
                "label": SYSTEM_LABELS[sys_name],
                "color": SYSTEM_COLORS[sys_name],
                "data": [
                    float(comparison_df[(comparison_df.scenario == sc) & (comparison_df.system == sys_name)][time_col].iloc[0])
                    if not comparison_df[(comparison_df.scenario == sc) & (comparison_df.system == sys_name)].empty else None
                    for sc in scenarios
                ],
            }
            for sys_name in systems
        ],
        "datasets_carbon": [
            {
                "label": SYSTEM_LABELS[sys_name],
                "color": SYSTEM_COLORS[sys_name],
                "data": [
                    float(comparison_df[(comparison_df.scenario == sc) & (comparison_df.system == sys_name)][carbon_col].iloc[0])
                    if not comparison_df[(comparison_df.scenario == sc) & (comparison_df.system == sys_name)].empty else None
                    for sc in scenarios
                ],
            }
            for sys_name in systems
        ],
    }

    cards = [
        {"label": "Total planned distance-time", "value": _fmt(metrics["total_planned_time_s"], " s", 0)},
        {"label": "Total actual (simulated) time", "value": _fmt(metrics["total_actual_time_s"], " s", 0)},
        {"label": "Total planned carbon", "value": _fmt(metrics["total_planned_carbon_kg"], " kg")},
        {"label": "Total actual (simulated) carbon", "value": _fmt(metrics["total_actual_carbon_kg"], " kg")},
        {"label": "Mean abs. time deviation", "value": _fmt((metrics["mean_abs_time_deviation_pct"] or 0) * 100, "%")},
        {
            "label": "EV feasibility rate",
            "value": _fmt((metrics["ev_feasibility_rate"] or 0) * 100, "%") if metrics["ev_feasibility_rate"] is not None else "N/A",
            "status": "good" if (metrics["ev_feasibility_rate"] or 0) >= 0.8 else ("critical" if metrics["ev_feasibility_rate"] is not None else None),
        },
        {"label": "Negotiation rounds used", "value": str(metrics["negotiation_rounds_used"])},
        {"label": "Carbon saved by optimizer", "value": _fmt(metrics["carbon_saved_by_optimizer_kg"], " kg")},
    ]

    data_json = json.dumps({
        "cards": cards,
        "summary_text": summary_text,
        "per_route": per_route,
        "comparison": comparison_chart_data,
        "ink": INK,
        "surface": SURFACE,
        "status": STATUS,
    })

    return _HTML_TEMPLATE.replace("__DASHBOARD_DATA__", data_json)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ESG Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
  :root {
    --surface-1: #fcfcfb;
    --page: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #898781;
    --grid: #e1e0d9;
    --border: rgba(11,11,11,0.10);
    --good: #0ca30c;
    --critical: #d03b3b;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --surface-1: #1a1a19;
      --page: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --grid: #2c2c2a;
      --border: rgba(255,255,255,0.10);
    }
  }
  :root[data-theme="dark"] {
    --surface-1: #1a1a19;
    --page: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --grid: #2c2c2a;
    --border: rgba(255,255,255,0.10);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background: var(--page);
    color: var(--text-primary);
    padding: 16px;
  }
  .wrap { max-width: 1080px; margin: 0 auto; }
  h1 { font-size: 22px; font-weight: 600; margin: 4px 0 4px; }
  .subtitle { color: var(--text-secondary); font-size: 14px; margin: 0 0 20px; }
  .cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
  }
  .card {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
  }
  .card .label { font-size: 12px; color: var(--text-secondary); margin-bottom: 6px; }
  .card .value { font-size: 24px; font-weight: 600; }
  .card .value.good { color: var(--good); }
  .card .value.critical { color: var(--critical); }
  .panel {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 20px;
  }
  .panel h2 { font-size: 15px; font-weight: 600; margin: 0 0 12px; }
  .narrative {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 20px;
    font-size: 15px;
    line-height: 1.55;
  }
  .narrative .tag {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-muted);
    margin-bottom: 8px;
  }
  canvas { max-height: 320px; }
  .charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 700px) { .charts-row { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="wrap">
  <h1>ESG Dashboard</h1>
  <p class="subtitle">Intelligent Decentralized Logistics Framework &mdash; Bengaluru</p>

  <div class="cards" id="cards"></div>

  <div class="narrative">
    <div class="tag">ESG Reporter summary</div>
    <div id="narrative-text"></div>
  </div>

  <div class="panel">
    <h2>Planned vs. simulated-actual time per route (mixed_priority scenario)</h2>
    <canvas id="routeTimeChart"></canvas>
  </div>

  <div class="charts-row">
    <div class="panel">
      <h2>Total planned time by system, per scenario</h2>
      <canvas id="systemTimeChart"></canvas>
    </div>
    <div class="panel">
      <h2>Total planned carbon by system, per scenario</h2>
      <canvas id="systemCarbonChart"></canvas>
    </div>
  </div>
</div>

<script>
const DATA = __DASHBOARD_DATA__;

function isDark() {
  const t = document.documentElement.getAttribute('data-theme');
  if (t === 'dark') return true;
  if (t === 'light') return false;
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}
const dark = isDark();
const ink = dark ? '#ffffff' : DATA.ink.primary;
const inkSecondary = dark ? '#c3c2b7' : DATA.ink.secondary;
const grid = dark ? '#2c2c2a' : DATA.surface.grid;

Chart.defaults.font.family = "system-ui, -apple-system, 'Segoe UI', sans-serif";
Chart.defaults.color = inkSecondary;
Chart.defaults.borderColor = grid;

// Stat cards
const cardsEl = document.getElementById('cards');
DATA.cards.forEach(c => {
  const div = document.createElement('div');
  div.className = 'card';
  const valueClass = c.status ? `value ${c.status}` : 'value';
  div.innerHTML = `<div class="label">${c.label}</div><div class="${valueClass}">${c.value}</div>`;
  cardsEl.appendChild(div);
});

document.getElementById('narrative-text').textContent = DATA.summary_text;

// Chart 1: per-route planned vs actual time
new Chart(document.getElementById('routeTimeChart'), {
  type: 'bar',
  data: {
    labels: DATA.per_route.map(r => r.request_id),
    datasets: [
      {
        label: 'Planned (s)',
        data: DATA.per_route.map(r => r.planned_time_s),
        backgroundColor: '#2a78d6',
        borderRadius: 4,
        categoryPercentage: 0.7,
        barPercentage: 0.9,
      },
      {
        label: 'Actual, simulated (s)',
        data: DATA.per_route.map(r => r.actual_time_s),
        backgroundColor: '#eb6834',
        borderRadius: 4,
        categoryPercentage: 0.7,
        barPercentage: 0.9,
      },
    ],
  },
  options: {
    responsive: true,
    plugins: { legend: { position: 'top', labels: { color: inkSecondary } } },
    scales: {
      x: { grid: { display: false }, ticks: { color: inkSecondary } },
      y: { grid: { color: grid }, ticks: { color: inkSecondary }, beginAtZero: true },
    },
  },
});

function systemComparisonChart(canvasId, datasets, yLabel) {
  new Chart(document.getElementById(canvasId), {
    type: 'bar',
    data: {
      labels: DATA.comparison.scenarios,
      datasets: datasets.map(d => ({
        label: d.label,
        data: d.data,
        backgroundColor: d.color,
        borderRadius: 4,
        categoryPercentage: 0.7,
        barPercentage: 0.85,
      })),
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'top', labels: { color: inkSecondary, boxWidth: 12 } } },
      scales: {
        x: { grid: { display: false }, ticks: { color: inkSecondary, maxRotation: 30, minRotation: 0 } },
        y: { grid: { color: grid }, ticks: { color: inkSecondary }, beginAtZero: true, title: { display: true, text: yLabel, color: inkSecondary } },
      },
    },
  });
}

systemComparisonChart('systemTimeChart', DATA.comparison.datasets_time, 'seconds');
systemComparisonChart('systemCarbonChart', DATA.comparison.datasets_carbon, 'kg CO2');
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════════
# CLI -- generates dashboard.html next to this file. Requires a working LLM
# key in .env (runs one full pipeline execution for the summary/narrative).
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import glob

    sys.stdout.reconfigure(encoding="utf-8")

    sample_costs_dir = Path(__file__).resolve().parent.parent / "sample_costs"
    csv_files = sorted(glob.glob(str(sample_costs_dir / "*.csv")))
    if not csv_files:
        raise SystemExit(f"No sample cost CSVs found in {sample_costs_dir}")
    costs_df = pd.read_csv(csv_files[0])

    results_dir = Path(__file__).resolve().parent / "evaluation_results"
    summary_path = results_dir / "summary_results.csv"
    if summary_path.exists():
        print(f"Using multi-snapshot comparison data from {summary_path}")
        comparison_df = pd.read_csv(summary_path)
    else:
        print("No multi-snapshot summary found -- running a fresh single-snapshot evaluation for the comparison chart.")
        comparison_df = run_evaluation(costs_df, seed=0)

    html = build_dashboard_html(costs_df, comparison_df, seed=0)

    out_path = Path(__file__).resolve().parent / "dashboard.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")
