"""
app_common.py

Pieces shared by the Streamlit pages: live_app.py (single delivery, the
closed loop, the algorithm comparison) and pages/1_Multi-stop_fleet_planner.py (several
deliveries per vehicle, OR-Tools). Defined once here so both pages share
one GNN, one road graph and one set of caches.
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
for _sub in ("gnn", "agents"):
    if str(ROOT / _sub) not in sys.path:
        sys.path.insert(0, str(ROOT / _sub))

import folium  # noqa: E402
import torch  # noqa: E402

from gnn_inference import GNNCostPredictor  # noqa: E402

CHECKPOINT_DIR = ROOT / "gnn_checkpoints_v3"
SNAPSHOT_DIR = ROOT / "dataset_kaggle" / "test"
GRAPH_DIR = ROOT / "bengaluru_graph"
SNAPSHOT_INDEX = SNAPSHOT_DIR / "_index.json"

EV_REFERENCE_KWH = 50.0  # ev_energy_pct is % of a 50kWh battery (CLAUDE.md 2.1)
BENGALURU_CENTER = (12.9716, 77.5946)


def missing_data() -> list:
    """Data files the pages need that are not present (they are not in git; see README)."""
    missing = [p for p in (CHECKPOINT_DIR / "best_model.pt", GRAPH_DIR / "nodes.geojson",
                           GRAPH_DIR / "edges.geojson") if not p.exists()]
    if not glob.glob(str(SNAPSHOT_DIR / "snapshot_*.pt")):
        missing.append(SNAPSHOT_DIR)
    return missing


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

    # Indexed by row (== costs_df row == edges.geojson feature), not by
    # (u, v): 83 junction pairs are joined by two parallel roads, and which
    # one a route uses is decided per snapshot by canonical_edges().
    edge_geom = [[(lat, lon) for lon, lat in f["geometry"]["coordinates"]] for f in edges]  # (lat, lon) for folium
    edge_len = [f["properties"].get("length") or 0.0 for f in edges]
    return nodes_df, edge_geom, edge_len


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


def nearest_node(nodes_df, lat, lon) -> int:
    dlat = np.radians(nodes_df["lat"].values - lat)
    dlon = np.radians(nodes_df["lon"].values - lon) * np.cos(np.radians(lat))
    return int(np.argmin(dlat ** 2 + dlon ** 2))


def node_label(nodes_df, idx) -> str:
    row = nodes_df.iloc[idx]
    name = row["name"] or f"Junction #{idx}"
    return f"{name} ({row.lat:.4f}, {row.lon:.4f})"


def draw_route(fmap, rows, edge_geom, color, label, weight=6, dash=None, opacity=0.9):
    coords = []
    for r in rows:
        seg = edge_geom[r]
        coords.extend(seg if not coords else seg[1:])
    if coords:
        folium.PolyLine(coords, color=color, weight=weight, opacity=opacity,
                        dash_array=dash, tooltip=label).add_to(fmap)
    return coords
