"""
pages/1_Multi-stop_fleet_planner.py

(Not named multi_stop.py: the pages/ folder is on the import path when the
page runs, and that name would shadow agents/multi_stop.py.)

Second page of the Streamlit app: several deliveries per vehicle, planned
with OR-Tools (agents/multi_stop.py, CLAUDE.md 12.4). Pick a depot, a few
drop-offs and a fleet of up to three vehicles; the page plans who takes
which stops and in what order, compares it with two simpler ways of doing
the same deliveries, and drives the plan through the Execution agent's
noise model to check the deadlines.

Run with the rest of the app:  streamlit run live_app.py
"""

import random
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import folium  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

from app_common import (BENGALURU_CENTER, load_graph, load_snapshot_index, missing_data,  # noqa: E402
                        node_label, predict_costs)
from multi_stop import SERVICE_TIME_S, plan_tours, simulate_tours  # noqa: E402
from routing_solver import RoutingSolver, canonical_edges  # noqa: E402
from schemas import DeliveryRequest, Vehicle  # noqa: E402
from vehicles import GRID_KG_CO2_PER_KWH, VEHICLE_TYPES  # noqa: E402

PRESETS = {
    "fastest": {"time": 0.7, "delay": 0.2, "carbon": 0.05, "ev": 0.05},
    "balanced": {"time": 0.25, "delay": 0.25, "carbon": 0.25, "ev": 0.25},
    "greenest": {"time": 0.1, "delay": 0.1, "carbon": 0.4, "ev": 0.4},
}
# Categorical slots 1-3 (light steps; the map tiles are light). Only three:
# on a map every tour can sit next to every other, and the palette's first
# three slots are the ones validated all-pairs (dataviz palette.md).
TOUR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
TYPE_BY_LABEL = {p.label: key for key, p in VEHICLE_TYPES.items()}
DEFAULT_FLEET = pd.DataFrame([
    {"Vehicle": VEHICLE_TYPES["ev_ldv"].label, "Battery (kWh)": 30.0, "Payload (kg)": 300.0},
    {"Vehicle": VEHICLE_TYPES["diesel_ldv"].label, "Battery (kWh)": None, "Payload (kg)": 300.0},
    {"Vehicle": VEHICLE_TYPES["cng_3w"].label, "Battery (kWh)": None, "Payload (kg)": 300.0},
])
METHOD_LABELS = {"optimised": "OR-Tools plan", "nearest_neighbour": "Nearest-neighbour heuristic",
                 "separate_trips": "Every delivery its own trip"}

st.set_page_config(page_title="Multi-stop fleet planner", layout="wide")
missing = missing_data()
if missing:
    st.error("Missing data files: " + ", ".join(str(m) for m in missing))
    st.stop()

nodes_df, edge_geom, edge_len = load_graph()
snap_df = load_snapshot_index()
snap_df["label"] = snap_df.apply(
    lambda r: f"{r.date} {r.hour:02d}:00 · {r.weather}" + (" · incident" if r.incident else ""), axis=1)
depots = nodes_df[(nodes_df["name"] != "") & (nodes_df["category"] != "ev_station")].sort_values("name")
depot_options = {row["name"]: idx for idx, row in depots.iterrows()}

ss = st.session_state
ss.setdefault("ms_plan", None)

# ---- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Plan a delivery round")
    depot_name = st.selectbox("Depot", list(depot_options), key="ms_depot")
    depot = depot_options[depot_name]
    snap_label = st.selectbox(f"Traffic snapshot ({len(snap_df)} available)", snap_df["label"], key="ms_snap",
                              index=int((snap_df["snapshot"] == "snapshot_07241").idxmax()))
    n_stops = st.slider("Drop-offs", 3, 15, 8, key="ms_stops")
    stop_seed = st.number_input("Which drop-offs (random seed)", 0, 999, 0, key="ms_seed",
                                help="Drop-offs are random junctions 10-40 minutes from the depot.")
    priority = st.selectbox("Priority", list(PRESETS), index=2, key="ms_priority")
    with_deadlines = st.toggle("Deadline on every other drop-off", value=True, key="ms_deadlines",
                               help="1.5 × the direct time from the depot + 20 minutes.")
    back = st.toggle("Vehicles return to the depot", value=True, key="ms_return")
    time_limit = st.slider("OR-Tools search time (s)", 1, 10, 3, key="ms_limit")
    st.markdown("**Fleet** (up to three vehicles)")
    fleet_df = st.data_editor(
        DEFAULT_FLEET, key="ms_fleet", num_rows="dynamic", hide_index=True, width="stretch",
        column_config={
            "Vehicle": st.column_config.SelectboxColumn(options=list(TYPE_BY_LABEL), required=True),
            "Battery (kWh)": st.column_config.NumberColumn(min_value=1.0, max_value=200.0,
                                                           help="Electric vehicles only"),
            "Payload (kg)": st.column_config.NumberColumn(min_value=5.0, max_value=20000.0, required=True),
        })
    go = st.button("🚚 Plan the round", type="primary", width="stretch")

snap_row = snap_df[snap_df["label"] == snap_label].iloc[0]
costs_df, infer_s = predict_costs(snap_row["file"])


def pick_stops(seed: int, count: int) -> list[DeliveryRequest]:
    solver = RoutingSolver(costs_df)
    rng = random.Random(seed)
    candidates = list(range(len(nodes_df)))
    stops, tries = [], 0
    while len(stops) < count and tries < 2000:
        tries += 1
        node = rng.choice(candidates)
        if node == depot or any(s.destination == node for s in stops):
            continue
        out, ret = solver.find_route(depot, node), solver.find_route(node, depot)
        if not (out["success"] and ret["success"]) or not 600 <= out["expected_travel_time_s"] <= 2400:
            continue
        deadline = 1.5 * out["expected_travel_time_s"] + 1200 if with_deadlines and len(stops) % 2 == 0 else None
        stops.append(DeliveryRequest(request_id=f"D{len(stops) + 1}", origin=depot, destination=node,
                                     package_weight_kg=float(rng.choice([10, 20, 40, 60, 80])), deadline_s=deadline))
    return stops


def fleet_from_editor(df: pd.DataFrame) -> list[Vehicle]:
    vehicles = []
    for i, row in df.dropna(subset=["Vehicle"]).head(3).reset_index(drop=True).iterrows():
        vtype = TYPE_BY_LABEL[row["Vehicle"]]
        is_ev = VEHICLE_TYPES[vtype].is_ev
        battery = row.get("Battery (kWh)")
        vehicles.append(Vehicle(
            vehicle_id=f"V{i + 1}", vehicle_type=vtype, current_location=depot,
            max_payload_kg=float(row["Payload (kg)"]) if pd.notna(row.get("Payload (kg)")) else None,
            battery_capacity_kwh=(float(battery) if pd.notna(battery) else 30.0) if is_ev else None))
    return vehicles


if go:
    vehicles = fleet_from_editor(fleet_df)
    if not vehicles:
        ss.ms_plan = {"error": "Add at least one vehicle."}
    else:
        with st.spinner("Choosing drop-offs, pricing every leg for each vehicle type and searching with OR-Tools..."):
            try:
                stops = pick_stops(int(stop_seed), int(n_stops))
                plan = plan_tours(costs_df, depot, stops, vehicles, weights=PRESETS[priority],
                                  time_limit_s=float(time_limit), return_to_depot=back)
                sims = simulate_tours(plan, costs_df, vehicles, seed=int(stop_seed))
                ss.ms_plan = {"plan": plan, "stops": stops, "vehicles": vehicles, "sims": sims, "depot": depot,
                              "priority": priority, "snapshot": snap_label}
            except Exception as e:
                ss.ms_plan = {"error": f"{type(e).__name__}: {str(e)[:500]}"}

# ---- Header ------------------------------------------------------------------
st.title("Multi-stop fleet planner")
st.caption(
    f"Live GNN v3 inference on **{snap_label}** ({infer_s * 1000:.0f} ms). Several deliveries per vehicle: OR-Tools "
    "decides which vehicle takes which drop-offs and in what order, within each vehicle's payload, battery and the "
    "drop-offs' deadlines. Each leg is the route the single-delivery planner would take, on that vehicle's own costs."
)

result = ss.ms_plan
if result and "error" in result:
    st.error(result["error"])
if not result or "error" in result:
    st.info("Choose a depot, the number of drop-offs and a fleet in the sidebar, then press **Plan the round**.")
    st.stop()

plan, stops, vehicles, sims = result["plan"], result["stops"], result["vehicles"], result["sims"]
o = plan["optimised"]

# ---- Comparison ------------------------------------------------------------------
st.subheader(f"The round — priority “{result['priority']}”, {len(stops)} drop-offs, {len(vehicles)} vehicles")
rows = {}
for key, label in METHOD_LABELS.items():
    r = plan[key]
    rows[label] = {
        "Driving + hand-over time (min)": r["total_time_s"] / 60,
        "CO₂ (kg)": r["total_co2_kg"],
        "Drop-offs served": f"{r['stops_served']} of {len(stops)}",
        "Deadlines met (expected)": f"{r['deadlines_met']} of {r['deadlines_total']}" if r["deadlines_total"] else "—",
    }
table = pd.DataFrame(rows).T
c1, c2 = st.columns([3, 2])
c1.dataframe(table.style.format({"Driving + hand-over time (min)": "{:.0f}", "CO₂ (kg)": "{:.2f}"}),
             width="stretch")
nn, sep = plan["nearest_neighbour"], plan["separate_trips"]
c2.markdown(
    f"- vs nearest-neighbour: **{o['total_time_s'] / 60 - nn['total_time_s'] / 60:+.0f} min**, "
    f"**{o['total_co2_kg'] - nn['total_co2_kg']:+.2f} kg CO₂**\n"
    f"- vs every delivery its own trip: **{o['total_time_s'] / 60 - sep['total_time_s'] / 60:+.0f} min**, "
    f"**{o['total_co2_kg'] - sep['total_co2_kg']:+.2f} kg CO₂**\n"
    f"- slowest vehicle back after **{o['makespan_s'] / 60:.0f} min**; vehicles used: **{o['vehicles_used']}**"
)
sim_stops = [s for t in sims for s in t["stops"] if s["deadline_s"] is not None]
if sim_stops:
    met = sum(1 for s in sim_stops if s["simulated_deadline_met"])
    (st.success if met == len(sim_stops) else st.warning)(
        f"Driven through the Execution agent's noise model, {met} of {len(sim_stops)} deadlines were met "
        f"(the plan expected {o['deadlines_met']} of {o['deadlines_total']}).")
if o["dropped"]:
    st.error("Could not be served within the payloads, batteries and deadlines: " + ", ".join(o["dropped"]))
st.caption(
    f"Times are expected values (typical delays included) plus {SERVICE_TIME_S // 60} minutes per hand-over. CO₂ is "
    f"each vehicle's own: tailpipe for combustion vehicles (India GHG Program, 2015), electricity × "
    f"{GRID_KG_CO2_PER_KWH} kg/kWh (CEA, FY 2023-24) for electric ones. OR-Tools minimises time and CO₂ weighted by "
    "the priority, summed over the vehicles; it does not try to finish sooner by spreading drop-offs across more "
    "vehicles unless deadlines require it. 'Every delivery its own trip' is how the single-delivery planner would do "
    "it: depot → drop-off → depot each time."
)

# ---- Map ---------------------------------------------------------------------------
ce = canonical_edges(costs_df)
row_of = dict(zip(zip(ce["src"].tolist(), ce["dst"].tolist()), ce.index.tolist()))
fmap = folium.Map(location=BENGALURU_CENTER, zoom_start=11, tiles="OpenStreetMap", control_scale=True)
bounds = []
for k, tour in enumerate(o["tours"]):
    if not tour["stops"]:
        continue
    color = TOUR_COLORS[k % len(TOUR_COLORS)]
    coords = []
    for u, v in zip(tour["path"][:-1], tour["path"][1:]):
        seg = edge_geom[row_of[(u, v)]] if (u, v) in row_of else []
        coords.extend(seg if not coords else seg[1:])
    if coords:
        folium.PolyLine(coords, color=color, weight=6, opacity=0.85,
                        tooltip=f"{tour['vehicle_id']} · {tour['vehicle_label']}").add_to(fmap)
        bounds += coords
    for n, s in enumerate(tour["stops"], start=1):
        row = nodes_df.iloc[s["node"]]
        late = s["deadline_met"] is False
        folium.Marker(
            (row.lat, row.lon),
            tooltip=(f"{s['request_id']} — {tour['vehicle_id']} stop {n}, arrives {s['arrival_s'] / 60:.0f} min"
                     + (f", deadline {s['deadline_s'] / 60:.0f} min" if s["deadline_s"] else "")),
            icon=folium.DivIcon(html=(
                f"<div style='background:{color};color:#fff;border:2px solid #fff;border-radius:50%;width:24px;"
                f"height:24px;line-height:20px;text-align:center;font:600 12px sans-serif;"
                f"{'outline:2px solid #d03b3b;' if late else ''}'>{n}</div>")),
        ).add_to(fmap)
for s in stops:
    if s.request_id in o["dropped"]:
        row = nodes_df.iloc[s.destination]
        folium.Marker((row.lat, row.lon), tooltip=f"{s.request_id} — not served",
                      icon=folium.Icon(color="gray", icon="ban", prefix="fa")).add_to(fmap)
d = nodes_df.iloc[result["depot"]]
folium.Marker((d.lat, d.lon), tooltip=f"Depot: {node_label(nodes_df, result['depot'])}",
              icon=folium.Icon(color="black", icon="warehouse", prefix="fa")).add_to(fmap)
bounds.append((d.lat, d.lon))
if len(bounds) >= 2:
    fmap.fit_bounds(bounds, padding=(30, 30))
st_folium(fmap, height=520, use_container_width=True, returned_objects=[], key="ms_map")
legend = " &nbsp; ".join(
    f"<span style='color:{TOUR_COLORS[k % 3]};font-weight:600'>━</span> {t['vehicle_id']} · {t['vehicle_label']}"
    for k, t in enumerate(o["tours"]) if t["stops"])
st.markdown(legend + " &nbsp; numbers = order of drop-offs", unsafe_allow_html=True)

# ---- Per vehicle ------------------------------------------------------------------
st.subheader("Each vehicle's round")
sim_by_vehicle = {s["vehicle_id"]: s for s in sims}
for k, tour in enumerate(o["tours"]):
    head = (f"{tour['vehicle_id']} · {tour['vehicle_label']} — {len(tour['stops'])} drop-offs, "
            f"{tour['time_s'] / 60:.0f} min, {tour['co2_kg']:.2f} kg CO₂, load {tour['load_kg']:.0f} of "
            f"{tour['max_payload_kg']:.0f} kg"
            + (f", {tour['energy_kwh']:.1f} of {tour['battery_kwh']:.0f} kWh" if tour["is_ev"] else ""))
    with st.expander(head, expanded=bool(tour["stops"])):
        if not tour["stops"]:
            st.write("Not used: the other vehicles cover the round at lower cost for this priority.")
            continue
        simulated = {s["request_id"]: s for s in sim_by_vehicle.get(tour["vehicle_id"], {"stops": []})["stops"]}
        st.dataframe(pd.DataFrame([{
            "Order": n, "Drop-off": s["request_id"], "Where": node_label(nodes_df, s["node"]),
            "Expected arrival (min)": s["arrival_s"] / 60,
            "Simulated arrival (min)": simulated.get(s["request_id"], {}).get("simulated_arrival_s", float("nan")) / 60,
            "Deadline (min)": s["deadline_s"] / 60 if s["deadline_s"] else None,
            "Deadline met (simulated)": ("—" if s["deadline_s"] is None else
                                         ("yes" if simulated.get(s["request_id"], {}).get("simulated_deadline_met")
                                          else "no")),
        } for n, s in enumerate(tour["stops"], start=1)]).style.format({
            "Expected arrival (min)": "{:.0f}", "Simulated arrival (min)": "{:.0f}", "Deadline (min)": "{:.0f}"}),
            width="stretch", hide_index=True)
