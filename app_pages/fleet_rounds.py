"""
app_pages/fleet_rounds.py  (the "Multi-stop fleet planner" page)

(Deliberately not named multi_stop.py: a page module with that name shadowed
agents/multi_stop.py when it lived in Streamlit's pages/ folder.)

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
import ui_theme as ui  # noqa: E402
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
TOUR_COLORS = [ui.tokens()["s_blue"], ui.tokens()["s_orange"], ui.tokens()["s_aqua"]]
TYPE_BY_LABEL = {p.label: key for key, p in VEHICLE_TYPES.items()}
DEFAULT_FLEET = pd.DataFrame([
    {"Vehicle": VEHICLE_TYPES["ev_ldv"].label, "Battery (kWh)": 30.0, "Payload (kg)": 300.0},
    {"Vehicle": VEHICLE_TYPES["diesel_ldv"].label, "Battery (kWh)": None, "Payload (kg)": 300.0},
    {"Vehicle": VEHICLE_TYPES["cng_3w"].label, "Battery (kWh)": None, "Payload (kg)": 300.0},
])
METHOD_LABELS = {"optimised": "OR-Tools plan", "nearest_neighbour": "Nearest-neighbour heuristic",
                 "separate_trips": "Every delivery its own trip"}

st.set_page_config(page_title="Multi-stop fleet · Bengaluru Logistics", page_icon=":material/local_shipping:", layout="wide")
ui.inject()
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
    ui.step(1, "Depot and traffic")
    depot_name = st.selectbox("Depot", list(depot_options), key="ms_depot")
    depot = depot_options[depot_name]
    snap_label = st.selectbox(f"Traffic snapshot ({len(snap_df)} available)", snap_df["label"], key="ms_snap",
                              index=int((snap_df["snapshot"] == "snapshot_07241").idxmax()))
    ui.step(2, "Drop-offs")
    n_stops = st.slider("Drop-offs", 3, 15, 8, key="ms_stops")
    stop_seed = st.number_input("Which drop-offs (random seed)", 0, 999, 0, key="ms_seed",
                                help="Drop-offs are random junctions 10-40 minutes from the depot.")
    priority = st.selectbox("Priority", list(PRESETS), index=2, key="ms_priority")
    with_deadlines = st.toggle("Deadline on every other drop-off", value=True, key="ms_deadlines",
                               help="1.5 × the direct time from the depot + 20 minutes.")
    back = st.toggle("Vehicles return to the depot", value=True, key="ms_return")
    time_limit = st.slider("OR-Tools search time (s)", 1, 10, 3, key="ms_limit")
    ui.step(3, "Fleet (up to three vehicles)")
    fleet_df = st.data_editor(
        DEFAULT_FLEET, key="ms_fleet", num_rows="dynamic", hide_index=True, width="stretch",
        column_config={
            "Vehicle": st.column_config.SelectboxColumn(options=list(TYPE_BY_LABEL), required=True),
            "Battery (kWh)": st.column_config.NumberColumn(min_value=1.0, max_value=200.0,
                                                           help="Electric vehicles only"),
            "Payload (kg)": st.column_config.NumberColumn(min_value=5.0, max_value=20000.0, required=True),
        })
    go = st.button("Plan the round", icon=":material/play_arrow:", type="primary", width="stretch")
    try:
        st.page_link("app_pages/single_delivery.py", label="Single delivery", icon=":material/alt_route:",
                     help="One trip from a source to a destination, disruptions on the road, the algorithm comparison")
    except Exception:  # page run on its own (not through live_app.py)
        pass

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
ui.header(
    "Multi-stop fleet planner",
    "Several deliveries per vehicle: OR-Tools decides which vehicle takes which drop-offs, and in what order, within each "
    "vehicle's payload, battery and the drop-offs' deadlines. Each leg is the route the single-delivery planner would take.",
    chips=[ui.chip("Snapshot", f"{snap_row['date']} {int(snap_row['hour']):02d}:00", "schedule", accent=True),
           ui.chip(str(snap_row["weather"]), symbol="cloud"),
           ui.chip("Scored", f"{len(costs_df):,} roads in {infer_s * 1000:.0f} ms", "bolt")],
    eyebrow="Bengaluru logistics")

result = ss.ms_plan
if result and "error" in result:
    st.error(result["error"])
if not result or "error" in result:
    st.markdown(
        '<div class="empty"><div class="t">Plan a delivery round in three steps</div><ol>'
        "<li>Choose a depot, the traffic snapshot and the number of drop-offs.</li>"
        "<li>Set the priority and up to three vehicles.</li>"
        "<li>Press <b>Plan the round</b>.</li></ol></div>", unsafe_allow_html=True)
    st.stop()

plan, stops, vehicles, sims = result["plan"], result["stops"], result["vehicles"], result["sims"]
o = plan["optimised"]
nn, sep = plan["nearest_neighbour"], plan["separate_trips"]
sim_stops = [s for t in sims for s in t["stops"] if s["deadline_s"] is not None]
met = sum(1 for s in sim_stops if s["simulated_deadline_met"])

# ---- The round at a glance ----------------------------------------------------------
st.markdown(ui.chip("Priority", result["priority"], accent=True) + " " + ui.chip("Drop-offs", str(len(stops))) + " "
            + ui.chip("Vehicles", str(len(vehicles))), unsafe_allow_html=True)
k = st.columns(4)
k[0].metric("Round time", f"{o['total_time_s'] / 60:.0f} min",
            f"{o['total_time_s'] / 60 - nn['total_time_s'] / 60:+.0f} min vs nearest-neighbour", delta_color="inverse",
            help="Driving plus hand-over time, summed over the vehicles. Expected values, typical delays included.")
k[1].metric("CO₂", f"{o['total_co2_kg']:.1f} kg", f"{o['total_co2_kg'] - nn['total_co2_kg']:+.1f} kg vs nearest-neighbour",
            delta_color="inverse", help="Each vehicle's own CO₂: tailpipe, or grid electricity for an EV.")
k[2].metric("Vehicles used", f"{o['vehicles_used']} of {len(vehicles)}",
            f"slowest back after {o['makespan_s'] / 60:.0f} min", delta_color="off", delta_arrow="off")
if sim_stops:
    k[3].metric("Deadlines met", f"{met} of {len(sim_stops)}", "in the simulated drive", delta_color="off", delta_arrow="off",
                help=f"The plan expected {o['deadlines_met']} of {o['deadlines_total']}.")
else:
    k[3].metric("Deadlines met", "—", "no deadlines set", delta_color="off", delta_arrow="off")
if sim_stops:
    (st.success if met == len(sim_stops) else st.warning)(
        f"Driven through the Execution agent's noise model, {met} of {len(sim_stops)} deadlines were met "
        f"(the plan expected {o['deadlines_met']} of {o['deadlines_total']}).",
        icon=":material/schedule:" if met == len(sim_stops) else ":material/warning:")
if o["dropped"]:
    st.error("Could not be served within the payloads, batteries and deadlines: " + ", ".join(o["dropped"]),
             icon=":material/block:")

# ---- Map ---------------------------------------------------------------------------
ce = canonical_edges(costs_df)
row_of = dict(zip(zip(ce["src"].tolist(), ce["dst"].tolist()), ce.index.tolist()))
fmap = ui.style_map(folium.Map(location=BENGALURU_CENTER, zoom_start=11, tiles="OpenStreetMap", control_scale=True))
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
            icon=ui.pin(str(n), "stop", color=color, flag=late),
        ).add_to(fmap)
for s in stops:
    if s.request_id in o["dropped"]:
        row = nodes_df.iloc[s.destination]
        folium.Marker((row.lat, row.lon), tooltip=f"{s.request_id} — not served",
                      icon=ui.pin("×", "stop", color=ui.tokens()["muted"])).add_to(fmap)
d = nodes_df.iloc[result["depot"]]
folium.Marker((d.lat, d.lon), tooltip=f"Depot: {node_label(nodes_df, result['depot'])}",
              icon=ui.pin("D", "dest")).add_to(fmap)
bounds.append((d.lat, d.lon))
if len(bounds) >= 2:
    fmap.fit_bounds(bounds, padding=(30, 30))
st_folium(fmap, height=520, use_container_width=True, returned_objects=[], key="ms_map")
ui.legend([(f"{t['vehicle_id']} · {t['vehicle_label']}", TOUR_COLORS[k % 3], "solid")
           for k, t in enumerate(o["tours"]) if t["stops"]] + [("Numbers show the order of drop-offs; D is the depot", ui.tokens()["muted"], "dot")])

# ---- How the plan compares ----------------------------------------------------------
st.subheader("How the plan compares")
METHOD_KEYS = ("optimised", "nearest_neighbour", "separate_trips")
MCOL = ui.route_colors()
ui.comparison_table(
    rows=[(METHOD_LABELS[key], [plan[key]["total_time_s"] / 60, plan[key]["total_co2_kg"], f"{plan[key]['stops_served']} of {len(stops)}",
                                 (f"{plan[key]['deadlines_met']} of {plan[key]['deadlines_total']}" if plan[key]["deadlines_total"] else "—")])
          for key in METHOD_KEYS],
    columns=[("Driving + hand-over (min)", "{:.0f}"), ("CO₂ (kg)", "{:.2f}"), ("Drop-offs served", "{}"),
             ("Deadlines met (expected)", "{}")],
    best={0: "min", 1: "min"},
    colours=[MCOL["multi_agent"], MCOL["baseline_time_only"], MCOL["baseline_carbon_only"]])
d_sep_t, d_sep_c = o["total_time_s"] / 60 - sep["total_time_s"] / 60, o["total_co2_kg"] - sep["total_co2_kg"]
ui.delta_chips([
    (f"Time vs every delivery its own trip {d_sep_t:+.0f} min", "good" if d_sep_t < 0 else "bad", "trending_down" if d_sep_t < 0 else "trending_up"),
    (f"CO₂ vs every delivery its own trip {d_sep_c:+.2f} kg", "good" if d_sep_c < 0 else "bad", "trending_down" if d_sep_c < 0 else "trending_up")])
with st.expander("Table view", icon=":material/table_rows:"):
    rows = {}
    for key in METHOD_KEYS:
        r = plan[key]
        rows[METHOD_LABELS[key]] = {
            "Driving + hand-over time (min)": r["total_time_s"] / 60, "CO₂ (kg)": r["total_co2_kg"],
            "Drop-offs served": f"{r['stops_served']} of {len(stops)}",
            "Deadlines met (expected)": f"{r['deadlines_met']} of {r['deadlines_total']}" if r["deadlines_total"] else "—"}
    st.dataframe(pd.DataFrame(rows).T.style.format({"Driving + hand-over time (min)": "{:.0f}", "CO₂ (kg)": "{:.2f}"}),
                 width="stretch")
with st.expander("How these numbers are computed", icon=":material/info:"):
    st.caption(
        f"Times are expected values (typical delays included) plus {SERVICE_TIME_S // 60} minutes per hand-over. CO₂ is "
        f"each vehicle's own: tailpipe for combustion vehicles (India GHG Program, 2015), electricity × "
        f"{GRID_KG_CO2_PER_KWH} kg/kWh (CEA, FY 2023-24) for electric ones. OR-Tools minimises time and CO₂ weighted by "
        "the priority, summed over the vehicles; it does not try to finish sooner by spreading drop-offs across more "
        "vehicles unless deadlines require it. 'Every delivery its own trip' is how the single-delivery planner would do "
        "it: depot → drop-off → depot each time.")

# ---- Per vehicle ------------------------------------------------------------------
st.subheader("Each vehicle's round")
sim_by_vehicle = {s["vehicle_id"]: s for s in sims}
for k, tour in enumerate(o["tours"]):
    head = (f"{tour['vehicle_id']} · {tour['vehicle_label']} — {len(tour['stops'])} drop-offs, "
            f"{tour['time_s'] / 60:.0f} min, {tour['co2_kg']:.2f} kg CO₂, load {tour['load_kg']:.0f} of "
            f"{tour['max_payload_kg']:.0f} kg"
            + (f", {tour['energy_kwh']:.1f} of {tour['battery_kwh']:.0f} kWh" if tour["is_ev"] else ""))
    with st.expander(head, expanded=bool(tour["stops"]), icon=":material/local_shipping:"):
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
            "Expected arrival (min)": "{:.0f}", "Simulated arrival (min)": "{:.0f}", "Deadline (min)": "{:.0f}"}, na_rep="—"),
            width="stretch", hide_index=True)
