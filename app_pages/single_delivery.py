"""
app_pages/single_delivery.py  (the "Single delivery: source -> destination"
page; until 2026-10-14 this was live_app.py, which is now only the entry
point that lists the pages)

Interactive, real-time version of the ESG dashboard (CLAUDE.md 4.3's
"upgrade to a small Streamlit app" follow-up). A user picks a source and a
destination -- on a map of Bengaluru or from the named depots/EV stations --
plus a traffic snapshot, a priority and a vehicle, and every click on
"Plan route" runs:

    live GNN inference (gnn/gnn_inference.py, v3 checkpoint)
      -> the full 6-agent pipeline (agents/pipeline.run_full_pipeline)
      -> map + metrics + agent trace + baseline comparison

and, for a planned route, "On the road" drives it through a change of
conditions part-way and shows the closed loop re-planning (or not):
agents/feedback_loop.py, CLAUDE.md 11.

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
import html
import json
import os
import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gnn"))
sys.path.insert(0, str(ROOT / "agents"))

import folium  # noqa: E402
import torch  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

from app_common import (BENGALURU_CENTER, CHECKPOINT_DIR, EV_REFERENCE_KWH, GRAPH_DIR, SNAPSHOT_DIR,  # noqa: E402
                        draw_route, load_graph, load_predictor, load_snapshot_index, missing_data, nearest_node,
                        node_label, predict_costs)
import ui_theme as ui  # noqa: E402
from disruptions import apply_incident, edge_at_fraction, edges_near  # noqa: E402
from routing_solver import DELAY_EXTRA_TIME_RANGE, RoutingSolver, canonical_edges  # noqa: E402
from schemas import DeliveryRequest, Vehicle  # noqa: E402
from snapshot_disruptions import with_congestion_surge, with_rain  # noqa: E402
from vehicles import GRID_KG_CO2_PER_KWH, VEHICLE_TYPES, costs_for_vehicle  # noqa: E402

# Same fixed slot order as agents/dashboard.py, so both dashboards read alike (blue is "our system");
# the steps follow the theme (ui_theme.TOKENS), so the map, table and charts always share one colour per route.
COLORS = ui.route_colors()
LABELS = {
    "multi_agent": "Multi-agent route",
    "baseline_time_only": "Fastest baseline (time only)",
    "baseline_carbon_only": "Greenest baseline (carbon only)",
}
BENCHMARK_CSV = ROOT / "gnn" / "benchmark_results" / "sssp_benchmark.csv"
# Categorical slots 1-3 of the same palette as COLORS below, in the step
# chosen for each theme (validated: all pairs pass in both modes).
CHART_SERIES = {
    "light": {"Grid graphs": "#2a78d6", "Sparse random graphs": "#eb6834", "Bengaluru road graph": "#1baf7a"},
    "dark": {"Grid graphs": "#3987e5", "Sparse random graphs": "#d95926", "Bengaluru road graph": "#199e70"},
}
ALGORITHM_OPTIONS = {"Dijkstra": "dijkstra", "Tsinghua BMSSP (Duan et al. 2025)": "bmssp"}
# Vehicle types (agents/vehicles.py). The first entry keeps the behaviour from
# before Phase 5: the GNN's own reference vehicle and the "Electric vehicle" toggle.
REFERENCE_VEHICLE = "GNN reference vehicle (as before)"
VEHICLE_OPTIONS = {REFERENCE_VEHICLE: None, **{p.label: key for key, p in VEHICLE_TYPES.items()}}
BASELINE_WEIGHTS = {
    "baseline_time_only": {"time": 1.0, "delay": 0.0, "carbon": 0.0, "ev": 0.0},
    "baseline_carbon_only": {"time": 0.0, "delay": 0.0, "carbon": 1.0, "ev": 0.0},
}

# "On the road": what can change after the vehicle leaves. The first is an
# assumed slowdown (the GNN cannot model incidents); the next three are GNN
# predictions on changed snapshot features. See agents/disruptions.py.
CHANGE_KINDS = {
    "A road incident ahead on the route": "incident",
    "Heavy rain starts across the city": "rain",
    "Traffic builds up on the route ahead": "surge",
    "The next traffic update of the day arrives": "update",
    "Nothing changes (control)": "none",
}
RULE_LABELS = {
    "same_route": "a fixed rule: the planned route is still the best one from here",
    "ev_range": "a fixed rule: the EV's remaining battery",
    "dominated": "a fixed rule: the new route is no faster and no cleaner",
    "small_gain": "a fixed rule: the gain is too small to turn a vehicle around for",
    "policy": "a fixed rule: the gain is above the 5% threshold (LLM is off)",
    "policy_fallback": "a fixed rule, because the LLM could not be reached",
    "llm": "the LLM, weighing the change against the delivery's priority",
    "deadline": "a fixed rule: the delivery's deadline",
    "not_informed": "nobody: the planner was never told",
    "arrived": "nobody: the vehicle had already arrived",
}
# Map ink for "On the road". The route in use keeps the multi-agent blue;
# what was driven and what was abandoned are neutral; red is the reserved
# "critical" status colour and always comes with a warning marker.
LOOP_COLORS = {"driven": ui.tokens()["ink2"], "taken": COLORS["multi_agent"], "not_taken": "#898781", "incident": "#d03b3b"}

# Same noise model as agents/execution_agent.py, used here only to turn the
# per-segment delay probabilities into numbers a user can act on. The route-
# level "P(at least one delayed segment)" is ~100% on every route (50-90
# segments at p~0.5 each), so it is not shown.
TIME_JITTER_PCT = 0.10  # mirrors agents/execution_agent.py; DELAY_EXTRA_TIME_RANGE comes from routing_solver

# One-click demo cases. Node ids are GNN node indices; snapshots are
# dataset_kaggle/test file stems. Numbers in the talking points were
# re-measured on 2026-10-11 after the routing fixes (CLAUDE.md 9) with the
# solver-only presets; "expected" means expected time including delays.
# `talk` is the speaker's note for the scenario. It is shown only as the hover tooltip (?) of
# the Quick start field (scenario_help); docs/review3_demo_script.md carries the same points.
DEMO_SCENARIOS = {
    "① Greener vs faster — Peenya → Rajajinagar, 9 AM": dict(
        src=5562, dst=1216, snapshot="snapshot_07241", priority="fastest", custom="", ev=True, battery=20.0, seed=0,
        talk="Plan with Priority = fastest, then greenest. Two different roads: fastest is 9.7 km, ~22.2 min "
             "expected, 2.95 kg CO₂; greenest is 7.0 km, ~25.2 min, 2.15 kg — 27% less CO₂ for about 3 minutes. "
             "With the LLM on and 'fastest', watch the Carbon Optimizer weigh that same trade (+3 min for "
             "−0.8 kg) and explain why it accepts or rejects it."),
    "② A green swap that isn't worth it — HSR Layout → Rajajinagar, 9 AM": dict(
        src=3133, dst=1216, snapshot="snapshot_07241", priority="fastest", custom="", ev=True, battery=20.0, seed=0,
        talk="17 km across the city. Fastest: ~45.4 min expected, 5.28 kg. Greenest: ~51.3 min, 5.11 kg — only 3% "
             "less CO₂ for 6 more minutes. Good case for showing that the system does not chase carbon blindly: "
             "compare the two priorities, and with the LLM on see how the Carbon Optimizer judges a small saving."),
    "③ Three priorities, three routes — Whitefield → Peenya, 9 AM": dict(
        src=1904, dst=5562, snapshot="snapshot_07241", priority="fastest", custom="", ev=True, battery=20.0, seed=0,
        talk="The longest trip (about 32 km). Switch Priority through fastest → balanced → greenest: three "
             "different routes with a steady trade-off — ~92.7 / 93.6 / 98.9 min expected and 10.08 / 9.83 / "
             "9.67 kg CO₂. Balanced gets most of the carbon saving for under a minute."),
    "④ Plain-English priority (LLM only) — Peenya → Rajajinagar, 9 AM": dict(
        src=5562, dst=1216, snapshot="snapshot_07241", priority="balanced",
        custom="Urgent medicine delivery, must arrive as fast as possible",
        ev=True, battery=20.0, seed=0,
        talk="Solver-only mode cannot read free text, so it falls back to equal 'balanced' weights and takes the "
             "slower, greener road (~25.2 min expected, 2.15 kg). With the LLM on, the Route Agent turns the "
             "sentence into time-heavy weights, explains why, and takes the fast road (~22.2 min, 2.95 kg). The "
             "exact weights vary a little run to run. Try your own sentence too."),
    "⑤ EV battery too small — Whitefield → Peenya, 6 kWh, 9 AM": dict(
        src=1904, dst=5562, snapshot="snapshot_07241", priority="fastest", custom="", ev=True, battery=6.0, seed=0,
        talk="The trip is expected to need about 13.7 kWh once typical delays are included (about 10 kWh "
             "delay-free). With the LLM on, the Coordinator detects the EV-range violation and negotiates for "
             "3 rounds (pushing the 'ev' weight up), but no route fits 6 kWh — it honestly reports 'not fully "
             "resolved'. Execution's hard rule then forces a re-plan. Set the battery to 20 kWh to show the "
             "feasible case."),
    "⑥ Same trip at 9 PM — Peenya → Rajajinagar, light traffic": dict(
        src=5562, dst=1216, snapshot="snapshot_07247", priority="fastest", custom="", ev=True, battery=20.0, seed=0,
        talk="Same trip as ①, but on the 21:00 snapshot: ~15.6 min expected instead of ~22.2 at 9 AM, because the "
             "GNN predicts far less delay at night. In both, the simulated drive lands within a few percent of "
             "the expected time, so the Execution agent has no reason to re-plan."),
}


# ══════════════════════════════════════════════════════════════════════════
# Cached loaders (the shared ones live in app_common.py)
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Re-scoring every road with the GNN on the new conditions...", max_entries=16)
def predict_changed(snapshot_file: str, kind: str, rows: tuple = ()):
    """GNN costs for a snapshot with rain, or with congestion surged on `rows`."""
    snapshot = torch.load(snapshot_file, weights_only=False)
    changed = with_rain(snapshot) if kind == "rain" else with_congestion_surge(snapshot, rows)
    return load_predictor().predict(changed)


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def route_distance_km(rows, edge_len) -> float:
    return sum(edge_len[r] for r in rows) / 1000.0


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


def route_summary(path, rows, total_time_s, total_carbon_kg, total_ev_pct, edge_len, edge_tp) -> dict:
    """rows: the costs_df / edges.geojson row of each edge along the path."""
    expected_s, p90_s = delay_outlook(path, edge_tp)
    return {
        "rows": rows,
        "distance_km": route_distance_km(rows, edge_len),
        "time_min": total_time_s / 60.0,
        "expected_min": expected_s / 60.0,
        "p90_min": p90_s / 60.0,
        "carbon_kg": total_carbon_kg,
        "ev_kwh": total_ev_pct / 100.0 * EV_REFERENCE_KWH,
        "edges": max(len(path) - 1, 0),
    }


def build_map(nodes_df, edge_geom, src, dst, plan):
    fmap = ui.style_map(folium.Map(location=BENGALURU_CENTER, zoom_start=11, tiles="OpenStreetMap",
                                   control_scale=True, zoom_control=True))

    # Named depots / EV stations as reference points (small, so the routes carry the colour).
    ring = "#ffffff"                                  # the basemap is light in both themes
    for idx, row in nodes_df[nodes_df["name"] != ""].iterrows():
        is_ev = row["category"] == "ev_station"
        folium.CircleMarker(
            (row.lat, row.lon), radius=5, weight=1.5,
            color=ring, fill=True, fill_color=ui.tokens()["s_aqua"] if is_ev else ui.tokens()["muted"], fill_opacity=1,
            tooltip=f"{row['name']} ({'EV station' if is_ev else 'depot'}, node {idx})",
        ).add_to(fmap)

    bounds = []
    if plan:
        for key in ("baseline_time_only", "baseline_carbon_only"):
            r = plan["routes"].get(key)
            if r and r["path"]:
                bounds += draw_route(fmap, r["rows"], edge_geom, COLORS[key], LABELS[key],
                                     weight=4, dash="6 8", opacity=0.75)
        r = plan["routes"].get("multi_agent")
        if r and r["path"]:
            bounds += draw_route(fmap, r["rows"], edge_geom, COLORS["multi_agent"],
                                 LABELS["multi_agent"], weight=7)

    for idx, kind, letter, label in ((src, "source", "A", "Source"), (dst, "dest", "B", "Destination")):
        if idx is not None:
            row = nodes_df.iloc[idx]
            folium.Marker((row.lat, row.lon), tooltip=f"{label}: {node_label(nodes_df, idx)}",
                          icon=ui.pin(letter, kind)).add_to(fmap)
            bounds.append((row.lat, row.lon))

    if len(bounds) >= 2:
        fmap.fit_bounds(bounds, padding=(30, 30))
    return fmap


# ══════════════════════════════════════════════════════════════════════════
# Planning
# ══════════════════════════════════════════════════════════════════════════

def plan_route(costs_df, src, dst, priority, is_ev, battery_kwh, payload_kg, use_agents, edge_len, seed,
               algorithm="dijkstra", vehicle_type=None, deadline_s=None):
    # Every number on the page is for the chosen vehicle: its own CO2
    # (tailpipe, or grid electricity for an EV) and, for an EV, its energy.
    vcosts = costs_for_vehicle(costs_df, vehicle_type)
    solver = RoutingSolver(vcosts, algorithm=algorithm)
    # Same choice among parallel roads as the solver and the Execution agent.
    ce = canonical_edges(vcosts)
    uv = list(zip(ce["src"].tolist(), ce["dst"].tolist()))
    row_of = dict(zip(uv, ce.index.tolist()))
    edge_tp = dict(zip(uv, zip(ce["travel_time_s"].tolist(), ce["delay_probability"].tolist())))
    routes, trace = {}, {}

    def summarize(path, total_time_s, total_carbon_kg, total_ev_pct):
        # total_ev_pct: pass the EXPECTED EV energy (delays included), the
        # figure the Coordinator plans range on.
        rows = [row_of[(u, v)] for u, v in zip(path[:-1], path[1:])]
        return {"path": path, **route_summary(path, rows, total_time_s, total_carbon_kg, total_ev_pct,
                                              edge_len, edge_tp)}

    # Baselines: the same plain-solver calls evaluation_harness.run_baseline uses.
    for key, weights in BASELINE_WEIGHTS.items():
        r = solver.find_route(src, dst, weights=weights)
        if not r["success"]:
            return {"error": r["reason"]}
        routes[key] = summarize(r["path"], r["total_travel_time_s"], r["total_carbon_kg"], r["expected_ev_energy_pct"])

    if use_agents:
        from pipeline import run_full_pipeline  # imports litellm; only needed in this mode

        request = DeliveryRequest(request_id="live-0", origin=src, destination=dst,
                                  priority=priority, package_weight_kg=min(payload_kg, 50.0), deadline_s=deadline_s)
        vehicle = Vehicle(vehicle_id="veh-live", is_ev=is_ev,
                          battery_capacity_kwh=battery_kwh if is_ev else None,
                          max_payload_kg=payload_kg, current_location=src, vehicle_type=vehicle_type)
        # The agents build their own RoutingSolver, which reads this setting.
        previous = os.environ.get("ROUTING_ALGORITHM")
        os.environ["ROUTING_ALGORITHM"] = algorithm
        t0 = time.perf_counter()
        try:
            out = run_full_pipeline([request], [vehicle], costs_df, seed=seed)
        finally:
            if previous is None:
                os.environ.pop("ROUTING_ALGORITHM", None)
            else:
                os.environ["ROUTING_ALGORITHM"] = previous
        trace["pipeline_s"] = time.perf_counter() - t0

        if out["unassigned"]:
            return {"error": "Dispatcher could not assign the vehicle (payload too small?)."}
        res = out["results"][0]
        if not res.success:
            return {"error": f"Route Agent failed: {res.reason}"}
        routes["multi_agent"] = summarize(res.path, res.total_travel_time_s, res.total_carbon_kg,
                                          res.expected_ev_energy_pct)
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
        trace["weights"] = weights
        trace["reasoning"] = "Solver-only mode: fixed preset weights, no LLM agents were called."
        if priority not in presets and priority != "balanced":
            trace["reasoning"] += (" Free-text priorities can't be interpreted without the LLM, "
                                   "so this fell back to 'balanced'.")
        if deadline_s is not None and r["success"] and r["expected_travel_time_s"] > deadline_s:
            # The Coordinator's deterministic deadline rule, without the LLM.
            from coordinator_agent import CoordinatorAgent
            from route_agent import build_route_result

            request = DeliveryRequest(request_id="live-0", origin=src, destination=dst, priority=priority,
                                      deadline_s=deadline_s, vehicle_id="veh-live")
            vehicle = Vehicle(vehicle_id="veh-live", vehicle_type=vehicle_type)
            met = CoordinatorAgent(costs_df, [vehicle]).meet_deadline(
                request, build_route_result(request, r, weights, None))
            r = {"path": met.path, "total_travel_time_s": met.total_travel_time_s,
                 "total_carbon_kg": met.total_carbon_kg, "expected_ev_energy_pct": met.expected_ev_energy_pct}
            trace["final_weights"] = met.weights_used
            trace["deadline_note"] = met.reasoning
        routes["multi_agent"] = summarize(r["path"], r["total_travel_time_s"], r["total_carbon_kg"],
                                          r["expected_ev_energy_pct"])

    # Same single-source search with both shortest-path algorithms, using the
    # weights the displayed route was planned with, so the page can show how
    # each one worked and that they agree (about 0.2-0.6 s per plan).
    trace["algorithms"] = solver.compare_algorithms(src, dst, trace.get("final_weights") or trace["weights"])

    return {"routes": routes, "trace": trace, "src": src, "dst": dst,
            "priority": priority, "is_ev": is_ev, "battery_kwh": battery_kwh,
            "use_agents": use_agents, "algorithm": algorithm, "seed": seed,
            "vehicle_type": vehicle_type, "deadline_s": deadline_s,
            "token": time.time()}  # ties an "On the road" result to the plan it was run on


def run_loop(plan, costs_df, snap_df, snap_row, kind, when, ahead, hops, slowdown, told):
    """
    Drives the planned route through a change of conditions `when` (0-1) of
    the way into the trip. Returns the feedback-loop report plus what the
    page needs to draw it.
    """
    from feedback_loop import ClosedLoopExecutor, fleet_learning  # imports litellm

    path = plan["routes"]["multi_agent"]["path"]
    weights = plan["trace"].get("final_weights") or plan["trace"]["weights"]
    vehicle_type = plan.get("vehicle_type")
    for_vehicle = lambda frame: costs_for_vehicle(frame, vehicle_type)  # noqa: E731
    gnn_costs, costs_df = costs_df, for_vehicle(costs_df)   # from here on, the vehicle's own costs
    ce = canonical_edges(costs_df)
    uv = list(zip(ce["src"].tolist(), ce["dst"].tolist()))
    edge_time = dict(zip(uv, ce["travel_time_s"].tolist()))
    row_of = dict(zip(uv, ce.index.tolist()))
    where = when + ahead * (1.0 - when)   # share of the whole trip

    blocked, basis = set(), "gnn"
    if kind == "incident":
        blocked = edges_near(costs_df, edge_at_fraction(path, edge_time, where), hops=hops)
        after = apply_incident(costs_df, blocked, slowdown=slowdown)
        event = ((f"A road incident is reported ahead on the planned route; traffic around it is about "
                  f"{slowdown:g} times slower.") if told else
                 (f"A road incident blocks part of the road ahead (traffic about {slowdown:g} times slower), "
                  "but nobody reports it to the planner."))
        basis = "assumed"
    elif kind == "rain":
        after = for_vehicle(predict_changed(snap_row["file"], "rain"))
        event = "Heavy rain has started across the city."
    elif kind == "surge":
        first = path.index(edge_at_fraction(path, edge_time, where)[0])
        rows = tuple(row_of[e] for e in zip(path[first:-1], path[first + 1:]))
        after = for_vehicle(predict_changed(snap_row["file"], "surge", rows))
        event = "Traffic has built up on the planned route ahead."
    elif kind == "update":
        later = snap_df[(snap_df["date"] == snap_row["date"]) & (snap_df["hour"] > snap_row["hour"])].sort_values("hour")
        if later.empty:
            return {"error": f"There is no later traffic snapshot on {snap_row['date']} (this is the last of the day). "
                             "Pick an earlier snapshot in the sidebar."}
        after = for_vehicle(predict_costs(later.iloc[0]["file"])[0])
        event = f"The {int(later.iloc[0]['hour']):02d}:00 traffic update has arrived."
    else:
        after, event, basis = costs_df, "Nothing has changed.", "none"

    vehicle = Vehicle(vehicle_id="veh-live", is_ev=plan["is_ev"],
                      battery_capacity_kwh=plan["battery_kwh"] if plan["is_ev"] else None)
    report = ClosedLoopExecutor(costs_df, after, seed=plan["seed"], use_llm=plan["use_agents"], informed=told).run(
        path, weights, priority=plan["priority"], vehicle=vehicle, disruption_after_fraction=when,
        request_id="live-0", event=event, deadline_s=plan.get("deadline_s"))

    fleet = None
    if not told and report["disruption_reached"]:
        learned = fleet_learning(costs_df, after, plan["src"], plan["dst"], weights, n_vehicles=5, seed=plan["seed"])
        fleet = {"trips": learned["trips"], "flagged": len(learned["store"].corrections())}

    def rows_of(p):
        return [row_of[e] for e in zip(p[:-1], p[1:]) if e in row_of]

    done = report["position"]["edges_done"]
    alternative = report.get("alternative")
    return {
        "report": report, "kind": kind, "event": event, "basis": basis, "told": told, "fleet": fleet,
        "slowdown": slowdown, "segments_scored": len(costs_df),
        "driven_rows": rows_of(path[:done + 1]),
        "planned_rest_rows": rows_of(path[done:]),
        "new_rows": rows_of(alternative["path"]) if alternative else [],
        "incident_rows": [row_of[e] for e in blocked if e in row_of],
    }


def build_loop_map(nodes_df, edge_geom, plan, loop):
    report = loop["report"]
    replanned = report["decision"]["replan"]
    fmap = ui.style_map(folium.Map(location=BENGALURU_CENTER, zoom_start=11, tiles="OpenStreetMap", control_scale=True))
    bounds = draw_route(fmap, loop["driven_rows"], edge_geom, LOOP_COLORS["driven"], "Driven before the change", weight=6)
    if replanned:
        bounds += draw_route(fmap, loop["planned_rest_rows"], edge_geom, LOOP_COLORS["not_taken"],
                             "Planned route, abandoned", weight=4, dash="6 8")
        bounds += draw_route(fmap, loop["new_rows"], edge_geom, LOOP_COLORS["taken"], "New route, taken", weight=7)
    else:
        if loop["new_rows"]:
            bounds += draw_route(fmap, loop["new_rows"], edge_geom, LOOP_COLORS["not_taken"],
                                 "New route, offered but not taken", weight=4, dash="6 8")
        bounds += draw_route(fmap, loop["planned_rest_rows"], edge_geom, LOOP_COLORS["taken"],
                             "Planned route, kept", weight=7)

    incident_points = []
    for r in loop["incident_rows"]:
        folium.PolyLine(edge_geom[r], color=LOOP_COLORS["incident"], weight=9, opacity=0.9,
                        tooltip=f"Incident: traffic {loop['slowdown']:g} times slower").add_to(fmap)
        incident_points += edge_geom[r]
    if incident_points:
        lat = sum(pt[0] for pt in incident_points) / len(incident_points)
        lon = sum(pt[1] for pt in incident_points) / len(incident_points)
        folium.Marker((lat, lon), tooltip=f"Incident: traffic {loop['slowdown']:g} times slower",
                      icon=ui.pin("!", "incident")).add_to(fmap)

    for idx, kind, letter, label in ((plan["src"], "source", "A", "Source"), (plan["dst"], "dest", "B", "Destination")):
        row = nodes_df.iloc[idx]
        folium.Marker((row.lat, row.lon), tooltip=f"{label}: {node_label(nodes_df, idx)}",
                      icon=ui.pin(letter, kind)).add_to(fmap)
        bounds.append((row.lat, row.lon))
    here = nodes_df.iloc[report["position"]["node"]]
    folium.Marker((here.lat, here.lon),
                  tooltip=f"The vehicle when conditions changed ({report['position']['elapsed_s'] / 60:.1f} min in)",
                  icon=ui.pin("V", "vehicle")).add_to(fmap)
    if len(bounds) >= 2:
        fmap.fit_bounds(bounds, padding=(30, 30))
    return fmap


# ══════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Route planner · Bengaluru Logistics", page_icon=":material/alt_route:", layout="wide")
ui.inject()

missing = missing_data()
if missing:
    st.error("Missing data files: " + ", ".join(str(m) for m in missing))
    st.stop()

nodes_df, edge_geom, edge_len = load_graph()
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
ss.setdefault("loop", None)  # last "On the road" result; only shown for the plan it was run on
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
ss.setdefault("algorithm", "Dijkstra")
ss.setdefault("vehicle_sel", REFERENCE_VEHICLE)
ss.setdefault("deadline_min", 0.0)
ss.setdefault("scenario", "—")


def load_scenario():
    sc = DEMO_SCENARIOS.get(ss.scenario)
    if not sc:
        return
    ss.src, ss.dst = sc["src"], sc["dst"]
    ss.snap_label = snap_label_by_name.get(sc["snapshot"], ss.snap_label)
    ss.priority_sel, ss.custom_text = sc["priority"], sc["custom"]
    ss.is_ev, ss.battery, ss.seed = sc["ev"], sc["battery"], sc["seed"]
    # The scenarios' talking points were measured with the reference vehicle and no deadline.
    ss.vehicle_sel, ss.deadline_min = REFERENCE_VEHICLE, 0.0
    ss.plan = None


# ---- Sidebar -------------------------------------------------------------
CIRCLED = str.maketrans("①②③④⑤⑥", "123456")


def scenario_label(key: str) -> str:
    """Display name of a demo scenario. The circled digits stay in the keys (the docs and the deck use them)."""
    if key == "—":
        return "None: set the trip yourself"
    if not any(c in key for c in "①②③④⑤⑥"):    # already a display label (idempotent, so applying it twice is harmless)
        return key
    number, _, rest = key.translate(CIRCLED).partition(" ")
    title, _, trip = rest.partition(" — ")
    return f"{number} · {title}: {trip}" if trip else f"{number} · {title}"


def scenario_help(key: str) -> str:
    """The (?) tooltip of the quick-start field: how it works, and the chosen scenario's talking points."""
    how = "Loads a prepared trip with every setting below. Press Plan route to run it."
    if key in DEMO_SCENARIOS:
        return f"**{scenario_label(key)}**\n\n{DEMO_SCENARIOS[key]['talk']}\n\n{how}"
    return how


with st.sidebar:
    st.selectbox("Quick start scenario", ["—"] + list(DEMO_SCENARIOS), key="scenario", on_change=load_scenario,
                 format_func=scenario_label, help=scenario_help(ss.scenario))

    ui.step(1, "Route")
    click_target = st.radio("A click on the map sets the…", ["Source", "Destination"],
                            horizontal=True)
    pick = st.selectbox("…or choose a named place", ["—"] + list(named_options),
                        key="named_pick")
    c1, c2 = st.columns(2)
    if c1.button("Set as source", width="stretch", disabled=pick == "—"):
        ss.src = named_options[pick]; ss.plan = None
    if c2.button("Set as destination", width="stretch", disabled=pick == "—"):
        ss.dst = named_options[pick]; ss.plan = None
    if st.button("Swap", icon=":material/swap_vert:", width="stretch"):
        ss.src, ss.dst = ss.dst, ss.src; ss.plan = None
    ui.ends(node_label(nodes_df, ss.src) if ss.src is not None else "—",
            node_label(nodes_df, ss.dst) if ss.dst is not None else "—")

    ui.step(2, "Traffic conditions")
    snap_label = st.selectbox(f"Traffic snapshot ({len(snap_df)} available)", snap_df["label"],
                              key="snap_label")
    snap_row = snap_df[snap_df["label"] == snap_label].iloc[0]

    ui.step(3, "Priority and vehicle")
    priority = st.selectbox("Priority", ["fastest", "balanced", "greenest"], key="priority_sel")
    custom = st.text_input("Or describe it in words (optional)", key="custom_text",
                           placeholder="e.g. urgent medicine, but avoid heavy traffic")
    if custom.strip():
        priority = custom.strip()
    vehicle_type = VEHICLE_OPTIONS[st.selectbox(
        "Vehicle", list(VEHICLE_OPTIONS), key="vehicle_sel",
        help="CO2 per km from the India GHG Program (2015); an electric vehicle's CO2 is its electricity x the "
             "CEA grid factor (0.727 kg/kWh, FY 2023-24). Travel time is the same for every vehicle.")]
    if vehicle_type is None:
        is_ev = st.toggle("Electric vehicle", key="is_ev")
    else:
        is_ev = VEHICLE_TYPES[vehicle_type].is_ev
    battery_kwh = st.slider("Battery capacity (kWh)", 1.0, 60.0, step=0.5, key="battery",
                            disabled=not is_ev)
    payload_kg = st.number_input("Max payload (kg)", 10.0, 5000.0, step=10.0, key="payload")
    deadline_min = st.number_input("Deadline, minutes after leaving (0 = none)", 0.0, 600.0, step=5.0,
                                   key="deadline_min",
                                   help="A hard limit: if the planned route is expected to arrive later, the "
                                        "planner moves toward faster weights until it fits.")

    ui.step(4, "Planner")
    use_agents = st.toggle("Use multi-agent LLM pipeline", key="use_agents",
                           help="Off = deterministic solver only (instant, no API calls).")
    seed = st.number_input("Simulation seed", 0, 10_000, step=1, key="seed",
                           help="Seed for the Execution agent's noise model.")
    algorithm = ALGORITHM_OPTIONS[st.selectbox(
        "Shortest-path algorithm", list(ALGORITHM_OPTIONS), key="algorithm",
        help="Both find a minimum-cost route. BMSSP is the Duan et al. (Tsinghua) algorithm, "
             "asymptotically faster than Dijkstra but slower at this graph size.")]
    go = st.button("Plan route", icon=":material/play_arrow:", type="primary", width="stretch",
                   disabled=ss.src is None or ss.dst is None or ss.src == ss.dst)
    try:
        st.page_link("app_pages/fleet_rounds.py", label="Fleet planner", icon=":material/local_shipping:",
                     help="Several drop-offs per vehicle")
    except Exception:  # page run on its own (not through live_app.py): no page list to link into
        pass

# ---- Run -----------------------------------------------------------------
costs_df, infer_s = predict_costs(snap_row["file"])

if go:
    with st.spinner("Running agents (Route → Coordinator → Carbon Optimizer → Execution → ESG)..."
                    if use_agents else "Solving..."):
        try:
            ss.plan = plan_route(costs_df, ss.src, ss.dst, priority, is_ev, battery_kwh,
                                 payload_kg, use_agents, edge_len, int(seed), algorithm,
                                 vehicle_type=vehicle_type, deadline_s=deadline_min * 60 if deadline_min > 0 else None)
            ss.plan["snapshot"] = snap_label
        except Exception as e:  # rate limits etc. -- show, don't crash the app
            ss.plan = {"error": f"{type(e).__name__}: {str(e)[:600]}"}

plan = ss.plan if ss.plan and "error" not in ss.plan else None
if plan and not plan["use_agents"]:
    LABELS = {**LABELS, "multi_agent": "Solver-only route (no LLM)"}

# ---- Header ---------------------------------------------------------------
ui.header(
    "Route planner",
    "A graph neural network scores every road in the network for time, delay risk, CO₂ and EV energy; "
    "six agents then plan, check and report the route.",
    chips=[ui.chip("Snapshot", f"{snap_row['date']} {int(snap_row['hour']):02d}:00", "schedule", accent=True),
           ui.chip(str(snap_row["weather"]), symbol="cloud"),
           ui.chip("Scored", f"{len(costs_df):,} roads in {infer_s * 1000:.0f} ms", "bolt")],
    eyebrow="Bengaluru logistics")
if ss.plan and "error" in ss.plan:
    st.error(ss.plan["error"])

# ---- Map (left) and the headline result (right) ------------------------------
if plan:
    routes, trace = plan["routes"], plan["trace"]
    ma = routes["multi_agent"]
    col_map, col_res = st.columns([3, 2], gap="large")
else:
    col_map, col_res = st.container(), None

with col_map:
    fmap = build_map(nodes_df, edge_geom, ss.src, ss.dst, plan)
    map_state = st_folium(fmap, height=560, use_container_width=True,
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

    ui.legend(([(LABELS["multi_agent"], COLORS["multi_agent"], "solid"),
                (LABELS["baseline_time_only"], COLORS["baseline_time_only"], "dash"),
                (LABELS["baseline_carbon_only"], COLORS["baseline_carbon_only"], "dash")] if plan else [])
              + [("EV station", ui.tokens()["s_aqua"], "dot"), ("Depot", ui.tokens()["muted"], "dot")])

if not plan:
    st.markdown(
        '<div class="empty"><div class="t">Plan a route in three steps</div><ol>'
        "<li>Choose a trip: click the map, pick named places, or load a quick start scenario.</li>"
        "<li>Choose the traffic snapshot, the priority and the vehicle.</li>"
        "<li>Press <b>Plan route</b>, then try a disruption under <b>On the road</b>.</li></ol></div>",
        unsafe_allow_html=True)
    st.stop()

# ---- Headline metrics ------------------------------------------------------
with col_res:
    st.markdown("### Planned route")
    st.markdown(
        ui.chip("Priority", plan["priority"], accent=True) + " "
        + ui.chip("Search", "Tsinghua BMSSP" if plan["algorithm"] == "bmssp" else "Dijkstra"),
        unsafe_allow_html=True)
    # Two columns of three rows: the tiles stay readable next to the map, and at laptop or split-screen width.
    m = st.columns(2) + st.columns(2) + st.columns(2)
    m[0].metric("Distance", f"{ma['distance_km']:.1f} km", f"{ma['edges']} segments", delta_color="off", delta_arrow="off")
    m[1].metric("Planned time", f"{ma['time_min']:.1f} min", help="GNN-predicted time with no delays.")
    m[2].metric("Expected with delays", f"{ma['expected_min']:.1f} min",
                f"+{ma['expected_min'] - ma['time_min']:.1f} min", delta_color="inverse",
                help="Planned time plus each segment's delay probability × average delay impact.")
    m[3].metric("90% arrive within", f"{ma['p90_min']:.1f} min",
                help="90th percentile of 2,000 simulated drives — a safe time to promise.")
    vtype = plan.get("vehicle_type")
    if vtype is None:
        m[4].metric("CO₂", f"{ma['carbon_kg']:.2f} kg", help="For the GNN's reference vehicle (0.27 kg CO₂/km in free flow).")
    elif VEHICLE_TYPES[vtype].is_ev:
        m[4].metric("CO₂ (grid electricity)", f"{ma['carbon_kg']:.2f} kg",
                    help=f"Electricity used × the CEA grid factor ({GRID_KG_CO2_PER_KWH} kg/kWh, FY 2023-24). No "
                         "tailpipe CO₂; transmission and charging losses not included.")
    else:
        m[4].metric("CO₂ (tailpipe)", f"{ma['carbon_kg']:.2f} kg",
                    help=f"{VEHICLE_TYPES[vtype].label}: {VEHICLE_TYPES[vtype].kg_co2_per_km} kg CO₂/km "
                         "(India GHG Program, 2015), with the GNN's congestion and weather effects.")
    if plan["is_ev"]:
        m[5].metric("EV energy", f"{ma['ev_kwh']:.2f} kWh", help="Expected battery use including typical delays.")
    else:
        m[5].metric("EV energy", "—", help="Not an electric vehicle.")

    if plan.get("deadline_s"):
        late = ma["expected_min"] > plan["deadline_s"] / 60
        (st.error if late else st.success)(
            f"Expected arrival {ma['expected_min']:.1f} min vs deadline {plan['deadline_s'] / 60:.0f} min — "
            + ("misses it even on the fastest route." if late else "within the deadline.")
            + (f" 90% of simulated drives arrive within {ma['p90_min']:.1f} min." if not late else ""),
            icon=":material/schedule:" if not late else ":material/alarm_off:")
        if trace.get("deadline_note"):
            st.caption(trace["deadline_note"])
    if plan["is_ev"]:
        ok = ma["ev_kwh"] <= plan["battery_kwh"]
        (st.success if ok else st.error)(
            f"Expected EV energy {ma['ev_kwh']:.2f} kWh (typical delays included) vs battery "
            f"{plan['battery_kwh']:.1f} kWh — "
            + ("within the battery." if ok else "more than the battery holds."),
            icon=":material/battery_charging_full:" if ok else ":material/battery_alert:")

    st.caption("Next: add a road incident, rain or a traffic build-up part-way through this trip and see whether the "
               "system re-plans, under [On the road](#on-the-road).")

# ---- Comparison vs baselines ----------------------------------------------
st.subheader("Compared with single-objective baselines")
cmp = pd.DataFrame({
    LABELS[k]: {
        "Distance (km)": r["distance_km"], "Planned (min)": r["time_min"],
        "With delays (min)": r["expected_min"], "90% within (min)": r["p90_min"],
        "CO₂ (kg)": r["carbon_kg"], "EV energy (kWh)": r["ev_kwh"],
    } for k, r in routes.items()
}).T.loc[[LABELS[k] for k in ("multi_agent", "baseline_time_only", "baseline_carbon_only")]]
ROUTE_ORDER = ("multi_agent", "baseline_time_only", "baseline_carbon_only")
ui.comparison_table(
    rows=[(LABELS[k], [routes[k]["distance_km"], routes[k]["time_min"], routes[k]["expected_min"], routes[k]["p90_min"],
                       routes[k]["carbon_kg"], routes[k]["ev_kwh"]]) for k in ROUTE_ORDER],
    columns=[("Distance (km)", "{:.2f}"), ("Planned (min)", "{:.1f}"), ("With delays (min)", "{:.1f}"),
             ("90% within (min)", "{:.1f}"), ("CO₂ (kg)", "{:.3f}"), ("EV energy (kWh)", "{:.3f}")],
    best={2: "min", 4: "min"},                       # the dot marks the lowest expected time and the lowest CO₂
    colours=[COLORS[k] for k in ROUTE_ORDER])
fast, green = routes["baseline_time_only"], routes["baseline_carbon_only"]


def delta_chip(label: str, value: float, unit: str, fmt: str, eps: float):
    if abs(value) < eps:
        return (f"{label}: no change", "info", "drag_handle")
    good = value < 0                                  # less time and less CO₂ are both better
    return (f"{label} {value:+{fmt}} {unit}", "good" if good else "bad", "trending_down" if good else "trending_up")


ui.delta_chips([
    delta_chip("Time vs fastest", ma["time_min"] - fast["time_min"], "min", ".1f", 0.05),
    delta_chip("CO₂ vs fastest", ma["carbon_kg"] - fast["carbon_kg"], "kg", ".3f", 0.0005),
    delta_chip("Time vs greenest", ma["time_min"] - green["time_min"], "min", ".1f", 0.05),
    delta_chip("CO₂ vs greenest", ma["carbon_kg"] - green["carbon_kg"], "kg", ".3f", 0.0005)])
b1, b2 = st.columns(2, gap="large")
with b1:
    ui.route_bar_chart([(LABELS[k], routes[k]["time_min"], COLORS[k]) for k in ROUTE_ORDER], "Planned time", "min", ".1f")
with b2:
    ui.route_bar_chart([(LABELS[k], routes[k]["carbon_kg"], COLORS[k]) for k in ROUTE_ORDER], "CO₂", "kg", ".2f")
with st.expander("Table view", icon=":material/table_rows:"):
    st.dataframe(cmp.style.format({"Distance (km)": "{:.2f}", "Planned (min)": "{:.1f}",
                                   "With delays (min)": "{:.1f}", "90% within (min)": "{:.1f}",
                                   "CO₂ (kg)": "{:.3f}", "EV energy (kWh)": "{:.3f}"}), width="stretch")

# ---- On the road: the closed loop -------------------------------------------
st.subheader("On the road: what if conditions change after the vehicle leaves?", anchor="on-the-road")
st.caption(
    "The route above was planned on one traffic snapshot. Here the vehicle drives it, conditions change part-way, "
    "and the Execution agent decides whether to re-plan from where the vehicle is. There is no live traffic feed "
    "in this project, so a change is a second set of road costs: either the GNN re-scoring the roads on changed "
    "conditions, or (for an incident, which the GNN cannot model) an assumed slowdown."
)
with st.form("loop_form"):
    f1, f2, f3 = st.columns([3, 2, 2])
    loop_kind = CHANGE_KINDS[f1.selectbox("What happens", list(CHANGE_KINDS))]
    loop_when = f2.slider("When, as a share of the trip driven", 10, 80, 30, step=5, format="%d%%")
    loop_told = f3.toggle("Tell the planner", value=True,
                          help="On: the planner gets the new conditions and may re-plan this trip. Off: nobody "
                               "tells it; the vehicle finishes its plan and the following vehicles have to learn "
                               "the problem from its telemetry.")
    g1, g2, g3 = st.columns(3)
    loop_ahead = g1.slider("How far ahead of the vehicle", 10, 90, 50, step=5, format="%d%%",
                           help="Incident and traffic build-up only: where it is, as a share of the trip still to drive.")
    loop_hops = g2.slider("Incident size (junctions around it)", 0, 3, 1,
                          help="Incident only. 0 = one road segment; each step adds the neighbouring junctions.")
    loop_slowdown = g3.slider("Incident slowdown", 2.0, 8.0, 4.0, step=0.5, format="x%.1f",
                              help="Incident only: how many times slower traffic is around it (an assumption).")
    drive_now = st.form_submit_button("Drive the trip", icon=":material/play_arrow:", type="primary")

if drive_now:
    with st.spinner("Driving the trip..."):
        try:
            ss.loop = run_loop(plan, costs_df, snap_df, snap_row, loop_kind, loop_when / 100, loop_ahead / 100,
                               int(loop_hops), float(loop_slowdown), loop_told)
        except Exception as e:  # rate limits etc. -- show, don't crash the app
            ss.loop = {"error": f"{type(e).__name__}: {str(e)[:600]}"}
        ss.loop["token"] = plan["token"]

loop = ss.loop if ss.loop and ss.loop.get("token") == plan["token"] else None
if loop and "error" in loop:
    st.warning(loop["error"])
elif loop:
    rep = loop["report"]
    decision, position = rep["decision"], rep["position"]
    promised_min = rep["promised"]["expected_time_s"] / 60
    outcome, other = rep["outcome"], rep["counterfactual"]

    if not rep["disruption_reached"]:
        st.info(f"The vehicle arrived after {outcome['actual_time_s'] / 60:.1f} min, before anything changed.")
    else:
        where_now = (f"{position['elapsed_s'] / 60:.1f} min into the trip, after {position['edges_done']} of "
                     f"{position['edges_total']} road segments")
        if decision["replan"]:
            st.success(f"**Re-planned.** {loop['event']} The vehicle was {where_now}.", icon=":material/alt_route:")
        else:
            st.info(f"**Kept the planned route.** {loop['event']} The vehicle was {where_now}.", icon=":material/straight:")
        ui.note(f"<b>Decided by</b> {html.escape(RULE_LABELS.get(decision['rule'], decision['rule']))}.<br>"
                f"{html.escape(str(decision['reasoning']))}", "gavel")

        kept = other if decision["replan"] else outcome   # what finishing the plan gives
        k = st.columns(4)
        k[0].metric("Promised at departure", f"{promised_min:.1f} min",
                    help="The plan's expected time including typical delays.")
        k[1].metric("Finishing the planned route", f"{kept['actual_time_s'] / 60:.1f} min",
                    f"{kept['actual_time_s'] / 60 - promised_min:+.1f} min vs promised", delta_color="inverse")
        if decision["replan"]:
            k[2].metric("With the re-plan", f"{outcome['actual_time_s'] / 60:.1f} min",
                        f"{-rep['saved']['time_s'] / 60:+.1f} min", delta_color="inverse")
            k[3].metric("CO₂ for the whole trip", f"{outcome['actual_carbon_kg']:.2f} kg",
                        f"{-rep['saved']['carbon_kg']:+.2f} kg", delta_color="inverse")
        else:
            if other:
                k[2].metric("Had it switched instead", f"{other['actual_time_s'] / 60:.1f} min",
                            f"{(other['actual_time_s'] - outcome['actual_time_s']) / 60:+.1f} min",
                            delta_color="inverse")
            k[3].metric("CO₂ for the whole trip", f"{outcome['actual_carbon_kg']:.2f} kg")

        notes = []
        if loop["basis"] == "assumed":
            notes.append(f"The incident is an assumption (traffic {loop['slowdown']:g} times slower on "
                         f"{len(loop['incident_rows'])} road segments), not a GNN prediction: the GNN cannot model "
                         "incidents.")
        elif loop["basis"] == "gnn":
            notes.append(f"The new conditions are GNN predictions: the snapshot was changed and all "
                         f"{loop['segments_scored']:,} road segments were scored again.")
        notes.append(f"On the new conditions the rest of the planned route is expected to take "
                     f"{rep['slowdown_ahead']:.2f} times as long as when it was planned.")
        if loop["kind"] == "surge" and rep["slowdown_ahead"] < 1.03:
            notes.append("Traffic on this route is already close to the maximum the data contains at this hour, so "
                         "a build-up changes almost nothing. Try a 13:00 or 21:00 snapshot.")
        if other:
            notes.append("Both arrival times are simulated drives with the same luck on every road they share, so "
                         "the difference between them is the effect of the decision.")
        st.caption(" ".join(notes))

        if plan["is_ev"]:
            used_kwh = outcome["actual_ev_energy_pct"] / 100 * EV_REFERENCE_KWH
            (st.error if outcome["ev_violation"] else st.success)(
                f"EV energy used on this drive: {used_kwh:.2f} kWh of a {plan['battery_kwh']:.1f} kWh battery"
                + (" — the battery ran out." if outcome["ev_violation"] else "."),
                icon=":material/battery_alert:" if outcome["ev_violation"] else ":material/battery_charging_full:")

        st_folium(build_loop_map(nodes_df, edge_geom, plan, loop), height=440, use_container_width=True,
                  returned_objects=[], key="loop_map")
        taken_label = "New route, taken" if decision["replan"] else "Planned route, kept"
        other_label = "Planned route, abandoned" if decision["replan"] else "New route, offered but not taken"
        ui.legend([("Driven before the change", LOOP_COLORS["driven"], "solid"), (taken_label, LOOP_COLORS["taken"], "solid"),
                   (other_label, LOOP_COLORS["not_taken"], "dash"), ("Incident (!)", LOOP_COLORS["incident"], "solid"),
                   ("Vehicle when conditions changed (V)", ui.tokens()["s_aqua"], "dot")])

    with st.expander("Telemetry: what the vehicle reported, segment by segment"):
        tel = pd.DataFrame(rep["telemetry"])
        tel["Segment"] = tel["src"].astype(str) + " → " + tel["dst"].astype(str)
        tel["Actual / expected"] = tel["actual_time_s"] / tel["expected_time_s"].where(tel["expected_time_s"] > 0)
        st.dataframe(
            tel.rename(columns={"phase": "When", "expected_time_s": "Expected (s)", "actual_time_s": "Actual (s)",
                                "delayed": "Delayed"})[
                ["When", "Segment", "Expected (s)", "Actual (s)", "Actual / expected", "Delayed"]
            ].style.format({"Expected (s)": "{:.1f}", "Actual (s)": "{:.1f}", "Actual / expected": "{:.2f}"}),
            width="stretch", hide_index=True)
        st.caption("Expected is what the planner believed at the time, typical delays included. This is the record "
                   "the long feedback loop learns from.")

    if loop["fleet"]:
        st.markdown("**Nobody told the planner — what the next vehicles on this same trip do**")
        trips = loop["fleet"]["trips"]
        st.dataframe(
            pd.DataFrame({
                "Vehicle": [t["vehicle"] for t in trips],
                "Road segments already flagged from telemetry": [t["segments_corrected_before_planning"] for t in trips],
                "Route": ["original plan" if t["same_as_no_feedback"]
                          else ("same as a fully informed planner" if t["same_as_informed"] else "a new route")
                          for t in trips],
                "Trip time (min)": [t["actual_time_s"] / 60 for t in trips],
                "Same drive without feedback (min)": [t["no_feedback_time_s"] / 60 for t in trips],
                "Same drive, planner fully informed (min)": [t["informed_time_s"] / 60 for t in trips],
            }).style.format({"Trip time (min)": "{:.1f}", "Same drive without feedback (min)": "{:.1f}",
                             "Same drive, planner fully informed (min)": "{:.1f}"}),
            width="stretch", hide_index=True)
        later = trips[1:]
        gained = sum(t["no_feedback_time_s"] - t["actual_time_s"] for t in later) / max(len(later), 1) / 60
        st.caption(
            "Each vehicle reports how long every segment actually took. A segment that takes far longer than "
            "predicted gets a correction factor, and the next vehicle is planned on the corrected costs. "
            f"After this run {loop['fleet']['flagged']} segments carry a correction; vehicles 2 to {len(trips)} "
            f"averaged {gained:+.1f} min against the same drives on the never-corrected plan"
            + (" (nothing was learned: the slowdown is inside what ordinary delays explain)."
               if not loop["fleet"]["flagged"] else "."))

# ---- Shortest-path algorithms ----------------------------------------------
st.subheader("Shortest-path search: Dijkstra vs Tsinghua BMSSP")
alg = trace["algorithms"]
dj, bm = alg["dijkstra"], alg["bmssp"]
if alg["same_path"]:
    verdict = f"Both algorithms found the **same route** ({dj['route_segments']} road segments)."
elif alg["same_cost"]:
    verdict = ("Both algorithms found routes of **exactly the same cost** but along different roads "
               "(a tie between equally good routes).")
else:
    verdict = "**The two algorithms disagree on this trip — that is a bug; please report it.**"
st.markdown(
    f"{verdict} That is expected: both are exact, so they must return a lowest-cost route. "
    "What differs is how they search and how much work it takes."
)
st.table(
    pd.DataFrame({
        "Dijkstra": {
            "Search time (ms)": f"{dj['ms']:.0f}",
            "Road segments examined": f"{dj['relaxations']:,}",
            "How it searches": "One sorted queue; always expands the single closest junction next",
            "Work behind the search": f"{dj['heap_pushes']:,} queue insertions",
            "Running time in theory": "m + n log n",
        },
        "Tsinghua BMSSP": {
            "Search time (ms)": f"{bm['ms']:.0f}",
            "Road segments examined": f"{bm['relaxations']:,}",
            "How it searches": f"Splits the search into bounded sub-problems, {bm['levels']} levels deep, "
                               "and carries only a few pivot junctions forward",
            "Work behind the search": f"{bm['pivot_searches']:,} pivot searches, {bm['queue_pulls']:,} queue pulls, "
                                      f"{bm['base_cases']:,} small Dijkstra runs",
            "Running time in theory": "m log^(2/3) n",
        },
    })
)
st.caption(
    f"Both searched from the source to all {alg['junctions_settled']:,} reachable junctions "
    f"({alg['junctions']:,} junctions and {alg['segments']:,} road segments in total), in plain Python, with the "
    "weights this route was planned with. On this trip the Tsinghua algorithm took "
    f"{bm['ms'] / max(dj['ms'], 1e-9):.0f} times as long and examined "
    f"{bm['relaxations'] / max(dj['relaxations'], 1):.1f} times as many road segments. Its advantage is "
    "theoretical: it grows more slowly on very large graphs."
)

if BENCHMARK_CSV.exists():
    bench = pd.read_csv(BENCHMARK_CSV)
    bench["family"] = bench["graph"].map(
        lambda g: "Bengaluru road graph" if g.startswith("Bengaluru")
        else ("Grid graphs" if g.startswith("grid") else "Sparse random graphs"))
    try:
        theme = st.context.theme.type or "dark"
    except Exception:  # older Streamlit without st.context.theme
        theme = "dark"
    series = CHART_SERIES.get(theme, CHART_SERIES["dark"])
    surface = st.get_option("theme.backgroundColor") or ("#0e1117" if theme == "dark" else "#ffffff")
    color = alt.Color("family:N", title=None,
                      scale=alt.Scale(domain=list(series), range=list(series.values())),
                      legend=alt.Legend(orient="top", symbolType="circle"))
    x = alt.X("n:Q", title="Junctions in the graph (log scale)", scale=alt.Scale(type="log"),
              axis=alt.Axis(format="~s", grid=False))
    y = alt.Y("time_ratio:Q", title="Times slower than Dijkstra", scale=alt.Scale(domain=[0, 24]))
    tooltip = [alt.Tooltip("graph:N", title="Graph"), alt.Tooltip("n:Q", title="Junctions", format=","),
               alt.Tooltip("dijkstra_ms:Q", title="Dijkstra (ms)", format=",.0f"),
               alt.Tooltip("bmssp_ms:Q", title="Tsinghua BMSSP (ms)", format=",.0f"),
               alt.Tooltip("time_ratio:Q", title="Times slower", format=".1f")]
    lines = alt.Chart(bench[bench["family"] != "Bengaluru road graph"]).mark_line(
        strokeWidth=2, strokeJoin="round", strokeCap="round").encode(x=x, y=y, color=color)
    points = alt.Chart(bench).mark_point(filled=True, size=90, opacity=1, stroke=surface, strokeWidth=2).encode(
        x=x, y=y, color=color, tooltip=tooltip)
    # Direct labels only where the story is: the grid line's two ends (the
    # gap narrowing) and this app's own graph. Both lines end at the same
    # value, so labelling every end would collide.
    grid = bench[bench["family"] == "Grid graphs"].sort_values("n")
    grid_ends = pd.concat([grid.head(1), grid.tail(1)]).assign(
        label=lambda d: d["time_ratio"].map(lambda r: f"{r:.1f} times slower"))
    blr = bench[bench["family"] == "Bengaluru road graph"]
    blr_label = pd.DataFrame({"n": [blr["n"].iloc[0]], "time_ratio": [blr["time_ratio"].max()],
                              "label": [f"this app's graph: {blr['time_ratio'].min():.0f} to {blr['time_ratio'].max():.0f}"]})
    # Labels wear a text colour, never the series colour (and not Altair's
    # default black, which disappears on the dark theme).
    ink = "#c3c2b7" if theme == "dark" else "#52514e"
    labels = (
        alt.Chart(grid_ends).mark_text(align="left", dx=8, dy=-12, fontSize=12, color=ink).encode(x=x, y=y, text="label:N")
        + alt.Chart(blr_label).mark_text(align="left", dx=10, dy=0, fontSize=12, color=ink).encode(x=x, y=y, text="label:N")
    )
    st.markdown("**How the gap changes with graph size** (from `gnn/benchmark_sssp.py`)")
    st.altair_chart((lines + points + labels).properties(height=300), width="stretch")
    st.caption(
        "Each point is one benchmark graph; hover for the timings. The Tsinghua algorithm is slower at every size "
        "tested, but on grid graphs the gap narrows steadily as the graph grows (11.4 times slower at 1,024 "
        "junctions, 7.0 at 262,144). The four Bengaluru points are the four weightings of this app's own graph. The "
        "largest sparse random graph was timed from a single source, so that point is the least reliable."
    )
    with st.expander("The benchmark numbers as a table"):
        table = bench[["graph", "n", "m", "dijkstra_ms", "bmssp_ms", "time_ratio", "relaxation_ratio"]].rename(columns={
            "graph": "Graph", "n": "Junctions", "m": "Road segments", "dijkstra_ms": "Dijkstra (ms)",
            "bmssp_ms": "Tsinghua BMSSP (ms)", "time_ratio": "Times slower", "relaxation_ratio": "Segments examined (ratio)"})
        st.dataframe(table.style.format({"Junctions": "{:,}", "Road segments": "{:,}", "Dijkstra (ms)": "{:,.0f}",
                                         "Tsinghua BMSSP (ms)": "{:,.0f}", "Times slower": "{:.1f}",
                                         "Segments examined (ratio)": "{:.2f}"}),
                     width="stretch", hide_index=True)

# ---- Agent trace -----------------------------------------------------------
st.subheader("What each agent did")
w = trace.get("weights", {})
with st.expander("1 · Route Agent: priority to solver weights", expanded=True, icon=":material/tune:"):
    if plan["use_agents"] and trace.get("rounds_used"):
        st.caption("These are the weights after the Coordinator's EV-range renegotiation (see step 2).")
    ui.weights_meter(w)
    st.caption(trace.get("reasoning") or "")
    if trace.get("swapped"):
        st.info("The Carbon Optimizer later replaced this route with a greener candidate using weights "
                f"{ {k: round(v, 3) for k, v in trace['final_weights'].items()} } (see step 3).",
                icon=":material/swap_horiz:")

if plan["use_agents"]:
    st.caption(f"Full pipeline took {trace['pipeline_s']:.1f} s.")
    with st.expander("2 · Coordinator: conflict negotiation", expanded=bool(trace["rounds_used"]), icon=":material/handshake:"):
        st.write(f"Negotiation rounds used: **{trace['rounds_used']}** · "
                 f"fully resolved: **{trace['fully_resolved']}**")
        if trace["rounds_used"]:
            st.caption("The EV range check failed, so the Coordinator asked the LLM for more "
                       "energy-efficient weights and re-routed. 'Fully resolved: False' means even the "
                       "most efficient route found still needs more energy than the battery holds.")
        else:
            st.caption("With a single delivery there is no shared-road conflict, and the EV range check "
                       "passed, so no negotiation was needed.")
    with st.expander("3 · Carbon Optimizer: greener swap proposals", expanded=True, icon=":material/eco:"):
        if trace["optimizer"]:
            for d in trace["optimizer"]:
                verdict = (ui.badge("Accepted", "good", "check_circle") if d["accepted"]
                           else ui.badge("Rejected", "bad", "cancel"))
                st.markdown(f"{verdict} &nbsp; Δ expected time **{d['time_delta_s']:+.0f} s**, "
                            f"ΔCO₂ **{d['carbon_delta_kg']:+.3f} kg**", unsafe_allow_html=True)
                st.caption(d.get("reasoning") or "")
        else:
            st.write("No candidates evaluated.")
    with st.expander("4 · Execution: simulated drive with noise", expanded=True, icon=":material/speed:"):
        for rep in trace["execution"]:
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Expected (with delays)", f"{rep['expected_time_s'] / 60:.1f} min",
                      help="What the plan promised: the GNN's time plus typical delays.")
            e2.metric("Simulated actual", f"{rep['actual_time_s'] / 60:.1f} min",
                      f"{rep['time_deviation_pct']:+.0%}", delta_color="inverse")
            e3.metric("Delayed segments", f"{rep['num_delayed_edges']}/{rep['num_edges']}")
            e4.metric("Re-plan?", "YES" if rep["replan_triggered"] else "no")
            if rep.get("hard_ev_violation"):
                st.error("Simulated EV energy exceeds the battery — hard re-plan.", icon=":material/battery_alert:")
            st.caption(rep.get("reasoning") or "Deviation within normal variance; no LLM review needed.")
        st.caption("This flag is raised after the whole simulated trip is over. To see a re-plan happen while the "
                   "vehicle is still driving, use “On the road” above.")
    with st.expander("5 · ESG Reporter", expanded=True, icon=":material/summarize:"):
        st.write(trace["esg"].get("summary") or "(narrative unavailable)")
        with st.popover("Raw ESG metrics", icon=":material/data_object:"):
            st.json(trace["esg"]["metrics"])
