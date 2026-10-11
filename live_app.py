"""
live_app.py

Interactive, real-time version of the ESG dashboard (CLAUDE.md 4.3's
"upgrade to a small Streamlit app" follow-up). A user picks a source and a
destination -- on a map of Bengaluru or from the named depots/EV stations --
plus a traffic snapshot, a priority and a vehicle, and every click on
"Plan route" runs:

    live GNN inference (gnn/gnn_inference.py, v3 checkpoint)
      -> the full 6-agent pipeline (agents/pipeline.run_full_pipeline)
      -> map + metrics + agent trace + baseline comparison

Nothing here re-implements agent logic; it only wires existing pieces to a UI.

Needs (not in git, see README section in CLAUDE.md / Drive):
    gnn_checkpoints_v3/{best_model.pt, normalizer.pt}
    dataset_kaggle/test/snapshot_*.pt
    bengaluru_graph/{nodes.geojson, edges.geojson}
    .env with LLM_MODEL + provider key (only for the multi-agent mode)

Run:
    streamlit run live_app.py
"""

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "gnn"))
sys.path.insert(0, str(ROOT / "agents"))

import folium  # noqa: E402
import torch  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

from gnn_inference import GNNCostPredictor  # noqa: E402
from routing_solver import RoutingSolver  # noqa: E402
from schemas import DeliveryRequest, Vehicle  # noqa: E402

CHECKPOINT_DIR = ROOT / "gnn_checkpoints_v3"
SNAPSHOT_DIR = ROOT / "dataset_kaggle" / "test"
GRAPH_DIR = ROOT / "bengaluru_graph"
SNAPSHOT_INDEX = SNAPSHOT_DIR / "_index.json"

EV_REFERENCE_KWH = 50.0  # ev_energy_pct is % of a 50kWh battery (CLAUDE.md 2.1)
BENGALURU_CENTER = (12.9716, 77.5946)

# Same fixed slot order as agents/dashboard.py, so both dashboards read alike.
COLORS = {
    "multi_agent": "#2a78d6",
    "baseline_time_only": "#eb6834",
    "baseline_carbon_only": "#1baf7a",
}
LABELS = {
    "multi_agent": "Multi-agent route",
    "baseline_time_only": "Fastest baseline (time only)",
    "baseline_carbon_only": "Greenest baseline (carbon only)",
}
BASELINE_WEIGHTS = {
    "baseline_time_only": {"time": 1.0, "delay": 0.0, "carbon": 0.0, "ev": 0.0},
    "baseline_carbon_only": {"time": 0.0, "delay": 0.0, "carbon": 1.0, "ev": 0.0},
}

# Same noise model as agents/execution_agent.py, used here only to turn the
# per-segment delay probabilities into numbers a user can act on. The route-
# level "P(at least one delayed segment)" is ~100% on every route (50-90
# segments at p~0.5 each), so it is not shown.
TIME_JITTER_PCT = 0.10
DELAY_EXTRA_TIME_RANGE = (1.3, 2.0)

# One-click demo cases, chosen by scanning all 156 named-place pairs on real
# GNN predictions for routes where priorities genuinely diverge. Node ids are
# GNN node indices; snapshots are dataset_kaggle/test file stems.
DEMO_SCENARIOS = {
    "① Greener vs faster — Peenya → Rajajinagar, 9 AM": dict(
        src=5562, dst=1216, snapshot="snapshot_07241", priority="fastest", custom="", ev=True, battery=20.0, seed=0,
        talk="Switch Priority between fastest → balanced → greenest and re-plan each time. Three different roads: "
             "~17.2 / 17.4 / 17.5 min and ~2.90 / 2.64 / 2.15 kg CO₂ — the greenest route cuts ~26% CO₂ for "
             "~20 s. With the LLM on, watch the Carbon Optimizer propose and accept a greener swap for 'fastest'."),
    "② Cross-city trip — HSR Layout EV station → Rajajinagar, 9 AM": dict(
        src=3133, dst=1216, snapshot="snapshot_07241", priority="fastest", custom="", ev=True, battery=20.0, seed=0,
        talk="18 km across the city. fastest ~36.2 min / 5.85 kg vs greenest ~36.6 min / 5.30 kg (−9% CO₂ for "
             "~25 s). The map shows the two routes taking different corridors."),
    "③ Agents repair a bad plan — Whitefield → Peenya, balanced, 9 AM": dict(
        src=1904, dst=5562, snapshot="snapshot_07241", priority="balanced", custom="", ev=True, battery=20.0, seed=0,
        talk="Run once WITHOUT the LLM: equal weights give a poor route (~86 min, ~12.1 kg). Then turn the LLM ON: "
             "the Carbon Optimizer proposes a greener candidate that turns out faster AND cleaner (~70 min, "
             "~10.3 kg) and accepts it."),
    "④ Plain-English priority (LLM only) — Peenya → Rajajinagar, 9 AM": dict(
        src=5562, dst=1216, snapshot="snapshot_07241", priority="balanced",
        custom="Eco-friendly delivery, the customer is happy to wait a little longer",
        ev=True, battery=20.0, seed=0,
        talk="Solver-only mode cannot read free text and falls back to equal 'balanced' weights (2.64 kg). With "
             "the LLM on, the Route Agent turns the sentence into carbon-heavy weights (carbon ~0.4, EV ~0.3-0.4) "
             "and explains why. The exact weights vary run to run, so the route lands on either the greenest "
             "road (~2.15 kg) or the balanced one (~2.64 kg) — point at the weights and reasoning in ①. Try your "
             "own sentence too, e.g. 'Urgent medicine, must arrive ASAP'."),
    "⑤ EV battery too small — Whitefield → Peenya, 6 kWh, 9 AM": dict(
        src=1904, dst=5562, snapshot="snapshot_07241", priority="fastest", custom="", ev=True, battery=6.0, seed=0,
        talk="The trip needs ~10-11 kWh. With the LLM on, the Coordinator detects the EV-range violation and "
             "negotiates for 3 rounds (pushing the 'ev' weight up), cuts energy use but cannot fit 6 kWh — it "
             "honestly reports 'not fully resolved'. Execution's hard rule then forces a re-plan. Set the battery "
             "to 20 kWh to show the feasible case."),
    "⑥ Same trip at 9 PM — Peenya → Rajajinagar, light traffic": dict(
        src=5562, dst=1216, snapshot="snapshot_07247", priority="greenest", custom="", ev=True, battery=20.0, seed=3,
        talk="Same trip as ①, but on the 21:00 snapshot: the GNN predicts ~13-14 min instead of ~17 (half the "
             "delay risk). The simulated drive runs only ~9-14% over plan, under the 15% threshold, so the "
             "Execution agent accepts it as normal variance — no LLM review, no re-plan (compare with ①)."),
}


# ══════════════════════════════════════════════════════════════════════════
# Cached loaders
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading GNN v3 checkpoint...")
def load_predictor():
    return GNNCostPredictor(checkpoint_dir=str(CHECKPOINT_DIR))


@st.cache_data(show_spinner="Loading Bengaluru road graph...")
def load_graph():
    """Node i in the GNN == feature i in nodes.geojson (verified: edge order matches too)."""
    nodes = json.loads((GRAPH_DIR / "nodes.geojson").read_text(encoding="utf-8"))["features"]
    edges = json.loads((GRAPH_DIR / "edges.geojson").read_text(encoding="utf-8"))["features"]

    osm_to_idx = {f["properties"]["osmid"]: i for i, f in enumerate(nodes)}
    nodes_df = pd.DataFrame({
        "lat": [f["properties"]["lat"] for f in nodes],
        "lon": [f["properties"]["lon"] for f in nodes],
        "name": [
            f["properties"]["warehouse_name"] or f["properties"]["ev_station_name"] or ""
            for f in nodes
        ],
        "category": [f["properties"]["node_category"] for f in nodes],
    })

    edge_geom, edge_len, edge_name = {}, {}, {}
    for f in edges:
        p = f["properties"]
        key = (osm_to_idx[p["u"]], osm_to_idx[p["v"]])
        # (lat, lon) order for folium
        edge_geom[key] = [(lat, lon) for lon, lat in f["geometry"]["coordinates"]]
        edge_len[key] = p.get("length") or 0.0
        edge_name[key] = p.get("road_name") or ""
    return nodes_df, edge_geom, edge_len, edge_name


@st.cache_data(show_spinner="Indexing traffic snapshots (first run only)...")
def load_snapshot_index():
    files = sorted(glob.glob(str(SNAPSHOT_DIR / "snapshot_*.pt")))
    cached = {}
    if SNAPSHOT_INDEX.exists():
        cached = json.loads(SNAPSHOT_INDEX.read_text())
    rows = []
    for f in files:
        name = Path(f).stem
        if name not in cached:
            s = torch.load(f, weights_only=False)
            cached[name] = {
                "date": str(s.date), "hour": int(s.hour),
                "weather": str(s.weather), "incident": bool(s.has_incident),
            }
        rows.append({"file": f, "snapshot": name, **cached[name]})
    SNAPSHOT_INDEX.write_text(json.dumps(cached))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Running live GNN inference on this snapshot...", max_entries=16)
def predict_costs(snapshot_file: str):
    snapshot = torch.load(snapshot_file, weights_only=False)
    t0 = time.perf_counter()
    costs = load_predictor().predict(snapshot)
    return costs, time.perf_counter() - t0


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def nearest_node(nodes_df, lat, lon) -> int:
    dlat = np.radians(nodes_df["lat"].values - lat)
    dlon = np.radians(nodes_df["lon"].values - lon) * np.cos(np.radians(lat))
    return int(np.argmin(dlat ** 2 + dlon ** 2))


def node_label(nodes_df, idx) -> str:
    row = nodes_df.iloc[idx]
    name = row["name"] or f"Junction #{idx}"
    return f"{name} ({row.lat:.4f}, {row.lon:.4f})"


def route_distance_km(path, edge_len) -> float:
    return sum(edge_len.get((u, v), 0.0) for u, v in zip(path[:-1], path[1:])) / 1000.0


def delay_outlook(path, edge_tp, n_samples=2000) -> tuple[float, float]:
    """
    (expected_time_s, p90_time_s) for a route under the Execution agent's
    noise model. Expected value is exact: jitter averages out, and a delayed
    segment adds mean(DELAY_EXTRA_TIME_RANGE) - 1 = 65% of its time. The
    90th percentile needs a quick Monte Carlo (fixed seed so it's stable).
    """
    tp = np.array([edge_tp[(u, v)] for u, v in zip(path[:-1], path[1:]) if (u, v) in edge_tp])
    if tp.size == 0:
        return 0.0, 0.0
    t, p = tp[:, 0], tp[:, 1]
    extra = (sum(DELAY_EXTRA_TIME_RANGE) / 2) - 1
    expected = float((t * (1 + p * extra)).sum())
    rng = np.random.default_rng(0)
    jitter = 1 + rng.uniform(-TIME_JITTER_PCT, TIME_JITTER_PCT, (n_samples, t.size))
    delayed = rng.random((n_samples, t.size)) < p
    mult = np.where(delayed, rng.uniform(*DELAY_EXTRA_TIME_RANGE, (n_samples, t.size)), 1.0)
    p90 = float(np.percentile((t * jitter * mult).sum(axis=1), 90))
    return expected, p90


def route_summary(path, total_time_s, total_carbon_kg, total_ev_pct, edge_len, edge_tp) -> dict:
    expected_s, p90_s = delay_outlook(path, edge_tp)
    return {
        "distance_km": route_distance_km(path, edge_len),
        "time_min": total_time_s / 60.0,
        "expected_min": expected_s / 60.0,
        "p90_min": p90_s / 60.0,
        "carbon_kg": total_carbon_kg,
        "ev_kwh": total_ev_pct / 100.0 * EV_REFERENCE_KWH,
        "edges": max(len(path) - 1, 0),
    }


def draw_route(fmap, path, edge_geom, color, label, weight=6, dash=None, opacity=0.9):
    coords = []
    for u, v in zip(path[:-1], path[1:]):
        seg = edge_geom.get((u, v))
        if seg:
            coords.extend(seg if not coords else seg[1:])
    if coords:
        folium.PolyLine(coords, color=color, weight=weight, opacity=opacity,
                        dash_array=dash, tooltip=label).add_to(fmap)
    return coords


def build_map(nodes_df, edge_geom, src, dst, plan):
    fmap = folium.Map(location=BENGALURU_CENTER, zoom_start=11, tiles="OpenStreetMap",
                      control_scale=True)

    # Named depots / EV stations as reference points.
    for idx, row in nodes_df[nodes_df["name"] != ""].iterrows():
        is_ev = row["category"] == "ev_station"
        folium.CircleMarker(
            (row.lat, row.lon), radius=5, weight=1,
            color="#1baf7a" if is_ev else "#52514e",
            fill=True, fill_opacity=0.8,
            tooltip=f"{row['name']} (node {idx})",
        ).add_to(fmap)

    bounds = []
    if plan:
        for key in ("baseline_time_only", "baseline_carbon_only"):
            r = plan["routes"].get(key)
            if r and r["path"]:
                bounds += draw_route(fmap, r["path"], edge_geom, COLORS[key], LABELS[key],
                                     weight=4, dash="6 8", opacity=0.75)
        r = plan["routes"].get("multi_agent")
        if r and r["path"]:
            bounds += draw_route(fmap, r["path"], edge_geom, COLORS["multi_agent"],
                                 LABELS["multi_agent"], weight=7)

    for idx, color, label in ((src, "green", "Source"), (dst, "red", "Destination")):
        if idx is not None:
            row = nodes_df.iloc[idx]
            folium.Marker((row.lat, row.lon), tooltip=f"{label}: {node_label(nodes_df, idx)}",
                          icon=folium.Icon(color=color, icon="flag" if label == "Destination" else "play")
                          ).add_to(fmap)
            bounds.append((row.lat, row.lon))

    if len(bounds) >= 2:
        fmap.fit_bounds(bounds, padding=(30, 30))
    return fmap


# ══════════════════════════════════════════════════════════════════════════
# Planning
# ══════════════════════════════════════════════════════════════════════════

def plan_route(costs_df, src, dst, priority, is_ev, battery_kwh, payload_kg, use_agents, edge_len, seed):
    solver = RoutingSolver(costs_df)
    edge_tp = dict(zip(zip(costs_df["src"], costs_df["dst"]),
                       zip(costs_df["travel_time_s"], costs_df["delay_probability"])))
    routes, trace = {}, {}

    # Baselines: the same plain-solver calls evaluation_harness.run_baseline uses.
    for key, weights in BASELINE_WEIGHTS.items():
        r = solver.find_route(src, dst, weights=weights)
        if not r["success"]:
            return {"error": r["reason"]}
        routes[key] = {"path": r["path"], **route_summary(
            r["path"], r["total_travel_time_s"], r["total_carbon_kg"],
            r["total_ev_energy_pct"], edge_len, edge_tp)}

    if use_agents:
        from pipeline import run_full_pipeline  # imports litellm; only needed in this mode

        request = DeliveryRequest(request_id="live-0", origin=src, destination=dst,
                                  priority=priority, package_weight_kg=min(payload_kg, 50.0))
        vehicle = Vehicle(vehicle_id="veh-live", is_ev=is_ev,
                          battery_capacity_kwh=battery_kwh if is_ev else None,
                          max_payload_kg=payload_kg, current_location=src)
        t0 = time.perf_counter()
        out = run_full_pipeline([request], [vehicle], costs_df, seed=seed)
        trace["pipeline_s"] = time.perf_counter() - t0

        if out["unassigned"]:
            return {"error": "Dispatcher could not assign the vehicle (payload too small?)."}
        res = out["results"][0]
        if not res.success:
            return {"error": f"Route Agent failed: {res.reason}"}
        routes["multi_agent"] = {"path": res.path, **route_summary(
            res.path, res.total_travel_time_s, res.total_carbon_kg,
            res.total_ev_energy_pct, edge_len, edge_tp)}
        # If the Carbon Optimizer swapped the route, res carries the
        # candidate's weights and no reasoning -- show the Route Agent's own
        # choice from the optimizer's audit fields instead.
        swap = next((d for d in out["decisions_log"] if d["accepted"]), None)
        trace.update({
            "weights": swap["original_weights"] if swap else res.weights_used,
            "reasoning": swap["original_reasoning"] if swap else res.reasoning,
            "final_weights": res.weights_used,
            "swapped": swap is not None,
            "rounds_used": out["rounds_used"],
            "fully_resolved": out["fully_resolved"],
            "optimizer": out["decisions_log"],
            "execution": out["execution_reports"],
            "esg": out["esg_report"],
        })
    else:
        presets = {
            "fastest": {"time": 0.7, "delay": 0.2, "carbon": 0.05, "ev": 0.05},
            "greenest": {"time": 0.1, "delay": 0.1, "carbon": 0.4, "ev": 0.4},
        }
        weights = presets.get(priority, {"time": 0.25, "delay": 0.25, "carbon": 0.25, "ev": 0.25})
        r = solver.find_route(src, dst, weights=weights)
        routes["multi_agent"] = {"path": r["path"], **route_summary(
            r["path"], r["total_travel_time_s"], r["total_carbon_kg"],
            r["total_ev_energy_pct"], edge_len, edge_tp)}
        trace["weights"] = weights
        trace["reasoning"] = "Solver-only mode: fixed preset weights, no LLM agents were called."
        if priority not in presets and priority != "balanced":
            trace["reasoning"] += (" Free-text priorities can't be interpreted without the LLM, "
                                   "so this fell back to 'balanced'.")

    return {"routes": routes, "trace": trace, "src": src, "dst": dst,
            "priority": priority, "is_ev": is_ev, "battery_kwh": battery_kwh,
            "use_agents": use_agents}


# ══════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Bengaluru Logistics – Live Planner", layout="wide")

missing = [p for p in (CHECKPOINT_DIR / "best_model.pt", GRAPH_DIR / "nodes.geojson",
                       GRAPH_DIR / "edges.geojson") if not p.exists()]
if missing or not glob.glob(str(SNAPSHOT_DIR / "snapshot_*.pt")):
    st.error("Missing data files: " + ", ".join(str(m) for m in missing or [SNAPSHOT_DIR]))
    st.stop()

nodes_df, edge_geom, edge_len, edge_name = load_graph()
snap_df = load_snapshot_index()
named = nodes_df[nodes_df["name"] != ""].sort_values("name")
named_options = {row["name"]: idx for idx, row in named.iterrows()}

snap_df["label"] = snap_df.apply(
    lambda r: f"{r.date} {r.hour:02d}:00 · {r.weather}" + (" · incident" if r.incident else ""),
    axis=1)
snap_label_by_name = dict(zip(snap_df["snapshot"], snap_df["label"]))

ss = st.session_state
ss.setdefault("src", named_options.get("Koramangala Depot"))
ss.setdefault("dst", named_options.get("Hebbal Logistics Park"))
ss.setdefault("plan", None)
ss.setdefault("last_click", None)
# Widget state lives under these keys (no value=/index= on the widgets), so a
# demo scenario can set all of them at once.
ss.setdefault("snap_label", snap_df["label"].iloc[0])
ss.setdefault("priority_sel", "balanced")
ss.setdefault("custom_text", "")
ss.setdefault("is_ev", True)
ss.setdefault("battery", 20.0)
ss.setdefault("payload", 200.0)
ss.setdefault("use_agents", True)
ss.setdefault("seed", 0)
ss.setdefault("scenario", "—")


def load_scenario():
    sc = DEMO_SCENARIOS.get(ss.scenario)
    if not sc:
        return
    ss.src, ss.dst = sc["src"], sc["dst"]
    ss.snap_label = snap_label_by_name.get(sc["snapshot"], ss.snap_label)
    ss.priority_sel, ss.custom_text = sc["priority"], sc["custom"]
    ss.is_ev, ss.battery, ss.seed = sc["ev"], sc["battery"], sc["seed"]
    ss.plan = None


# ---- Sidebar -------------------------------------------------------------
with st.sidebar:
    st.header("Plan a delivery")

    st.selectbox("🎬 Demo scenario (loads all settings below)", ["—"] + list(DEMO_SCENARIOS),
                 key="scenario", on_change=load_scenario)

    st.subheader("1 · Source & destination")
    click_target = st.radio("A click on the map sets the…", ["Source", "Destination"],
                            horizontal=True)
    pick = st.selectbox("…or choose a named place", ["—"] + list(named_options),
                        key="named_pick")
    c1, c2 = st.columns(2)
    if c1.button("Set as source", width="stretch", disabled=pick == "—"):
        ss.src = named_options[pick]; ss.plan = None
    if c2.button("Set as destination", width="stretch", disabled=pick == "—"):
        ss.dst = named_options[pick]; ss.plan = None
    if st.button("↔ Swap", width="stretch"):
        ss.src, ss.dst = ss.dst, ss.src; ss.plan = None
    st.caption(f"**Source:** {node_label(nodes_df, ss.src) if ss.src is not None else '—'}")
    st.caption(f"**Destination:** {node_label(nodes_df, ss.dst) if ss.dst is not None else '—'}")

    st.subheader("2 · Traffic conditions")
    snap_label = st.selectbox(f"Traffic snapshot ({len(snap_df)} available)", snap_df["label"],
                              key="snap_label")
    snap_row = snap_df[snap_df["label"] == snap_label].iloc[0]

    st.subheader("3 · Priority & vehicle")
    priority = st.selectbox("Priority", ["fastest", "balanced", "greenest"], key="priority_sel")
    custom = st.text_input("Or describe it in words (optional)", key="custom_text",
                           placeholder="e.g. urgent medicine, but avoid heavy traffic")
    if custom.strip():
        priority = custom.strip()
    is_ev = st.toggle("Electric vehicle", key="is_ev")
    battery_kwh = st.slider("Battery capacity (kWh)", 1.0, 60.0, step=0.5, key="battery",
                            disabled=not is_ev)
    payload_kg = st.number_input("Max payload (kg)", 10.0, 5000.0, step=10.0, key="payload")

    st.subheader("4 · Planner")
    use_agents = st.toggle("Use multi-agent LLM pipeline", key="use_agents",
                           help="Off = deterministic solver only (instant, no API calls).")
    seed = st.number_input("Simulation seed", 0, 10_000, step=1, key="seed",
                           help="Seed for the Execution agent's noise model.")
    go = st.button("🚚 Plan route", type="primary", width="stretch",
                   disabled=ss.src is None or ss.dst is None or ss.src == ss.dst)

# ---- Run -----------------------------------------------------------------
costs_df, infer_s = predict_costs(snap_row["file"])

if go:
    with st.spinner("Running agents (Route → Coordinator → Carbon Optimizer → Execution → ESG)..."
                    if use_agents else "Solving..."):
        try:
            ss.plan = plan_route(costs_df, ss.src, ss.dst, priority, is_ev, battery_kwh,
                                 payload_kg, use_agents, edge_len, int(seed))
            ss.plan["snapshot"] = snap_label
        except Exception as e:  # rate limits etc. -- show, don't crash the app
            ss.plan = {"error": f"{type(e).__name__}: {str(e)[:600]}"}

plan = ss.plan if ss.plan and "error" not in ss.plan else None
if plan and not plan["use_agents"]:
    LABELS = {**LABELS, "multi_agent": "Solver-only route (no LLM)"}

# ---- Header ---------------------------------------------------------------
st.title("Bengaluru Logistics — Live Route Planner")
st.caption(
    f"Live GNN v3 inference on **{snap_label}** — {len(costs_df):,} road segments scored in "
    f"{infer_s * 1000:.0f} ms. Pick points, then press **Plan route**."
)
if ss.plan and "error" in ss.plan:
    st.error(ss.plan["error"])
if ss.scenario in DEMO_SCENARIOS:
    st.info(f"**{ss.scenario}** — {DEMO_SCENARIOS[ss.scenario]['talk']}", icon="🎬")

# ---- Map ------------------------------------------------------------------
fmap = build_map(nodes_df, edge_geom, ss.src, ss.dst, plan)
map_state = st_folium(fmap, height=520, use_container_width=True,
                      returned_objects=["last_clicked"], key="map")
click = (map_state or {}).get("last_clicked")
if click and click != ss.last_click:
    ss.last_click = click
    idx = nearest_node(nodes_df, click["lat"], click["lng"])
    if click_target == "Source":
        ss.src = idx
    else:
        ss.dst = idx
    ss.plan = None
    st.rerun()

legend = " &nbsp; ".join(
    f"<span style='color:{COLORS[k]};font-weight:600'>━</span> {LABELS[k]}" for k in COLORS)
st.markdown(legend + " &nbsp; <span style='color:#1baf7a'>●</span> EV station"
                     " &nbsp; <span style='color:#52514e'>●</span> depot", unsafe_allow_html=True)

if not plan:
    st.info("Click the map (or pick named places in the sidebar) to set a source and destination, "
            "then press **Plan route**.")
    st.stop()

routes, trace = plan["routes"], plan["trace"]
ma = routes["multi_agent"]

# ---- Headline metrics ------------------------------------------------------
st.subheader(f"Planned route — priority: “{plan['priority']}”")
m = st.columns(6)
m[0].metric("Distance", f"{ma['distance_km']:.1f} km", f"{ma['edges']} road segments", delta_color="off")
m[1].metric("Planned time", f"{ma['time_min']:.1f} min", help="GNN-predicted time with no delays.")
m[2].metric("Expected with delays", f"{ma['expected_min']:.1f} min",
            f"+{ma['expected_min'] - ma['time_min']:.1f} min", delta_color="inverse",
            help="Planned time plus each segment's delay probability × average delay impact.")
m[3].metric("90% arrive within", f"{ma['p90_min']:.1f} min",
            help="90th percentile of 2,000 simulated drives — a safe time to promise.")
m[4].metric("CO₂", f"{ma['carbon_kg']:.2f} kg")
m[5].metric("EV energy", f"{ma['ev_kwh']:.2f} kWh")

if plan["is_ev"]:
    ok = ma["ev_kwh"] <= plan["battery_kwh"]
    (st.success if ok else st.error)(
        f"Planned EV energy {ma['ev_kwh']:.2f} kWh vs battery {plan['battery_kwh']:.1f} kWh — "
        + ("feasible on plan." if ok else "exceeds the battery even before simulated delays."))

# ---- Comparison vs baselines ----------------------------------------------
st.subheader("Compared with single-objective baselines")
cmp = pd.DataFrame({
    LABELS[k]: {
        "Distance (km)": r["distance_km"], "Planned (min)": r["time_min"],
        "With delays (min)": r["expected_min"], "90% within (min)": r["p90_min"],
        "CO₂ (kg)": r["carbon_kg"], "EV energy (kWh)": r["ev_kwh"],
    } for k, r in routes.items()
}).T.loc[[LABELS[k] for k in ("multi_agent", "baseline_time_only", "baseline_carbon_only")]]
c1, c2 = st.columns([3, 2])
c1.dataframe(cmp.style.format({"Distance (km)": "{:.2f}", "Planned (min)": "{:.1f}",
                               "With delays (min)": "{:.1f}", "90% within (min)": "{:.1f}",
                               "CO₂ (kg)": "{:.3f}", "EV energy (kWh)": "{:.3f}"}),
             width="stretch")
fast, green = routes["baseline_time_only"], routes["baseline_carbon_only"]
c2.markdown(
    f"- vs fastest baseline: **{ma['time_min'] - fast['time_min']:+.1f} min**, "
    f"**{ma['carbon_kg'] - fast['carbon_kg']:+.3f} kg CO₂**\n"
    f"- vs greenest baseline: **{ma['time_min'] - green['time_min']:+.1f} min**, "
    f"**{ma['carbon_kg'] - green['carbon_kg']:+.3f} kg CO₂**"
)
chart_df = pd.DataFrame({
    "system": [LABELS[k] for k in routes],
    "Time (min)": [r["time_min"] for r in routes.values()],
    "CO₂ (kg)": [r["carbon_kg"] for r in routes.values()],
}).set_index("system")
b1, b2 = st.columns(2)
b1.bar_chart(chart_df[["Time (min)"]], horizontal=True, color=COLORS["multi_agent"])
b2.bar_chart(chart_df[["CO₂ (kg)"]], horizontal=True, color=COLORS["baseline_carbon_only"])

# ---- Agent trace -----------------------------------------------------------
st.subheader("What each agent did")
w = trace.get("weights", {})
with st.expander("① Route Agent — priority → solver weights", expanded=True):
    if plan["use_agents"] and trace.get("rounds_used"):
        st.caption("These are the weights after the Coordinator's EV-range renegotiation (see ②).")
    st.write({k: round(v, 3) for k, v in w.items()})
    st.caption(trace.get("reasoning") or "")
    if trace.get("swapped"):
        st.info("The Carbon Optimizer later replaced this route with a greener candidate using weights "
                f"{ {k: round(v, 3) for k, v in trace['final_weights'].items()} } — see ③.")

if plan["use_agents"]:
    st.caption(f"Full pipeline took {trace['pipeline_s']:.1f} s.")
    with st.expander("② Coordinator — conflict negotiation", expanded=bool(trace["rounds_used"])):
        st.write(f"Negotiation rounds used: **{trace['rounds_used']}** · "
                 f"fully resolved: **{trace['fully_resolved']}**")
        if trace["rounds_used"]:
            st.caption("The EV range check failed, so the Coordinator asked the LLM for more "
                       "energy-efficient weights and re-routed. 'Fully resolved: False' means even the "
                       "most efficient route found still needs more energy than the battery holds.")
        else:
            st.caption("With a single delivery there is no shared-road conflict, and the EV range check "
                       "passed, so no negotiation was needed.")
    with st.expander("③ Carbon Optimizer — greener swap proposals", expanded=True):
        if trace["optimizer"]:
            for d in trace["optimizer"]:
                verdict = "✅ accepted" if d["accepted"] else "❌ rejected"
                st.markdown(f"{verdict} — Δtime **{d['time_delta_s']:+.0f} s**, "
                            f"ΔCO₂ **{d['carbon_delta_kg']:+.3f} kg**")
                st.caption(d.get("reasoning") or "")
        else:
            st.write("No candidates evaluated.")
    with st.expander("④ Execution — simulated drive with noise", expanded=True):
        for rep in trace["execution"]:
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Planned", f"{rep['planned_time_s'] / 60:.1f} min")
            e2.metric("Simulated actual", f"{rep['actual_time_s'] / 60:.1f} min",
                      f"{rep['time_deviation_pct']:+.0%}", delta_color="inverse")
            e3.metric("Delayed segments", f"{rep['num_delayed_edges']}/{rep['num_edges']}")
            e4.metric("Re-plan?", "YES" if rep["replan_triggered"] else "no")
            if rep.get("hard_ev_violation"):
                st.error("Simulated EV energy exceeds the battery — hard re-plan.")
            st.caption(rep.get("reasoning") or "Deviation within normal variance; no LLM review needed.")
    with st.expander("⑤ ESG Reporter", expanded=True):
        st.write(trace["esg"].get("summary") or "(narrative unavailable)")
        with st.popover("Raw ESG metrics"):
            st.json(trace["esg"]["metrics"])
