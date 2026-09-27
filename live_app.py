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


def route_summary(path, total_time_s, total_carbon_kg, total_ev_pct, delay_p, edge_len) -> dict:
    return {
        "distance_km": route_distance_km(path, edge_len),
        "time_min": total_time_s / 60.0,
        "carbon_kg": total_carbon_kg,
        "ev_kwh": total_ev_pct / 100.0 * EV_REFERENCE_KWH,
        "delay_p": delay_p,
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
    routes, trace = {}, {}

    # Baselines: the same plain-solver calls evaluation_harness.run_baseline uses.
    for key, weights in BASELINE_WEIGHTS.items():
        r = solver.find_route(src, dst, weights=weights)
        if not r["success"]:
            return {"error": r["reason"]}
        routes[key] = {"path": r["path"], **route_summary(
            r["path"], r["total_travel_time_s"], r["total_carbon_kg"],
            r["total_ev_energy_pct"], r["route_delay_probability"], edge_len)}

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
            res.total_ev_energy_pct, res.route_delay_probability, edge_len)}
        trace.update({
            "weights": res.weights_used,
            "reasoning": res.reasoning,
            "rounds_used": out["rounds_used"],
            "fully_resolved": out["fully_resolved"],
            "optimizer": out["decisions_log"],
            "execution": out["execution_reports"],
            "esg": out["esg_report"],
        })
    else:
        weights = {
            "fastest": {"time": 0.7, "delay": 0.2, "carbon": 0.05, "ev": 0.05},
            "greenest": {"time": 0.1, "delay": 0.1, "carbon": 0.4, "ev": 0.4},
        }.get(priority, {"time": 0.25, "delay": 0.25, "carbon": 0.25, "ev": 0.25})
        r = solver.find_route(src, dst, weights=weights)
        routes["multi_agent"] = {"path": r["path"], **route_summary(
            r["path"], r["total_travel_time_s"], r["total_carbon_kg"],
            r["total_ev_energy_pct"], r["route_delay_probability"], edge_len)}
        trace["weights"] = weights
        trace["reasoning"] = "Solver-only mode: fixed preset weights, no LLM agents were called."

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

ss = st.session_state
ss.setdefault("src", named_options.get("Koramangala Depot"))
ss.setdefault("dst", named_options.get("Hebbal Logistics Park"))
ss.setdefault("plan", None)
ss.setdefault("last_click", None)

# ---- Sidebar -------------------------------------------------------------
with st.sidebar:
    st.header("Plan a delivery")

    st.subheader("1 · Source & destination")
    click_target = st.radio("A click on the map sets the…", ["Source", "Destination"],
                            horizontal=True)
    pick = st.selectbox("…or choose a named place", ["—"] + list(named_options),
                        key="named_pick")
    c1, c2 = st.columns(2)
    if c1.button("Set as source", use_container_width=True, disabled=pick == "—"):
        ss.src = named_options[pick]; ss.plan = None
    if c2.button("Set as destination", use_container_width=True, disabled=pick == "—"):
        ss.dst = named_options[pick]; ss.plan = None
    if st.button("↔ Swap", use_container_width=True):
        ss.src, ss.dst = ss.dst, ss.src; ss.plan = None
    st.caption(f"**Source:** {node_label(nodes_df, ss.src) if ss.src is not None else '—'}")
    st.caption(f"**Destination:** {node_label(nodes_df, ss.dst) if ss.dst is not None else '—'}")

    st.subheader("2 · Traffic conditions")
    snap_df["label"] = snap_df.apply(
        lambda r: f"{r.date} {r.hour:02d}:00 · {r.weather}" + (" · incident" if r.incident else ""),
        axis=1)
    snap_label = st.selectbox(f"Traffic snapshot ({len(snap_df)} available)", snap_df["label"],
                              index=0)
    snap_row = snap_df[snap_df["label"] == snap_label].iloc[0]

    st.subheader("3 · Priority & vehicle")
    priority = st.selectbox("Priority", ["fastest", "balanced", "greenest"], index=1)
    custom = st.text_input("Or describe it in words (optional)",
                           placeholder="e.g. urgent medicine, but avoid heavy traffic")
    if custom.strip():
        priority = custom.strip()
    is_ev = st.toggle("Electric vehicle", value=True)
    battery_kwh = st.slider("Battery capacity (kWh)", 1.0, 60.0, 20.0, 0.5, disabled=not is_ev)
    payload_kg = st.number_input("Max payload (kg)", 10.0, 5000.0, 200.0, 10.0)

    st.subheader("4 · Planner")
    use_agents = st.toggle("Use multi-agent LLM pipeline", value=True,
                           help="Off = deterministic solver only (instant, no API calls).")
    seed = st.number_input("Simulation seed", 0, 10_000, 0, 1,
                           help="Seed for the Execution agent's noise model.")
    go = st.button("🚚 Plan route", type="primary", use_container_width=True,
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

# ---- Header ---------------------------------------------------------------
st.title("Bengaluru Logistics — Live Route Planner")
st.caption(
    f"Live GNN v3 inference on **{snap_label}** — {len(costs_df):,} road segments scored in "
    f"{infer_s * 1000:.0f} ms. Pick points, then press **Plan route**."
)
if ss.plan and "error" in ss.plan:
    st.error(ss.plan["error"])

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
m[0].metric("Distance", f"{ma['distance_km']:.1f} km")
m[1].metric("Travel time", f"{ma['time_min']:.1f} min")
m[2].metric("CO₂", f"{ma['carbon_kg']:.2f} kg")
m[3].metric("EV energy", f"{ma['ev_kwh']:.2f} kWh")
m[4].metric("P(any delay)", f"{ma['delay_p']:.0%}")
m[5].metric("Road segments", f"{ma['edges']}")

if plan["is_ev"]:
    ok = ma["ev_kwh"] <= plan["battery_kwh"]
    (st.success if ok else st.error)(
        f"Planned EV energy {ma['ev_kwh']:.2f} kWh vs battery {plan['battery_kwh']:.1f} kWh — "
        + ("feasible on plan." if ok else "exceeds the battery even before simulated delays."))

# ---- Comparison vs baselines ----------------------------------------------
st.subheader("Compared with single-objective baselines")
cmp = pd.DataFrame({
    LABELS[k]: {
        "Distance (km)": r["distance_km"], "Time (min)": r["time_min"],
        "CO₂ (kg)": r["carbon_kg"], "EV energy (kWh)": r["ev_kwh"], "P(any delay)": r["delay_p"],
    } for k, r in routes.items()
}).T.loc[[LABELS[k] for k in ("multi_agent", "baseline_time_only", "baseline_carbon_only")]]
c1, c2 = st.columns([3, 2])
c1.dataframe(cmp.style.format({"Distance (km)": "{:.2f}", "Time (min)": "{:.1f}", "CO₂ (kg)": "{:.3f}",
                               "EV energy (kWh)": "{:.3f}", "P(any delay)": "{:.0%}"}),
             use_container_width=True)
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
    st.write({k: round(v, 3) for k, v in w.items()})
    st.caption(trace.get("reasoning") or "")

if plan["use_agents"]:
    st.caption(f"Full pipeline took {trace['pipeline_s']:.1f} s.")
    with st.expander("② Coordinator — conflict negotiation"):
        st.write(f"Negotiation rounds used: **{trace['rounds_used']}** · "
                 f"fully resolved: **{trace['fully_resolved']}**")
        st.caption("With a single delivery there is no shared-road conflict; rounds are only "
                   "used if the EV range check fails.")
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
