"""
multi_stop.py

Several deliveries per vehicle (Phase 5 of the 2026-10-11 plan, CLAUDE.md
12.4): a capacitated vehicle routing problem with time windows and battery
limits, solved with Google OR-Tools. This is the "OR-Tools CVRP with carbon +
EV constraints" item from the Review 1 deck, at small scale (tens of stops).

The rest of the pipeline gives one delivery to one vehicle. Here a few
vehicles leave one depot and share many drop-offs, and OR-Tools decides
which vehicle takes which stops and in what order.

HOW IT IS BUILT
---------------
1. Legs. For every vehicle type in the fleet and every pair of points
   (depot + stops), the leg is the route the single-trip planner would take
   between them: RoutingSolver on that vehicle type's costs
   (agents/vehicles.py) with the delivery's weights. Each leg carries its
   expected time (delays included), that vehicle's CO2, and battery energy.
   One one-to-all search per point and vehicle type.
2. Objective, per vehicle: time and CO2 in real units, weighted by the
   delivery's priority ("time" + "delay" weight on expected time, "carbon"
   + "ev" weight on CO2), each divided by its average leg value so the two
   are comparable. Because CO2 is the vehicle's own, a mixed fleet is
   steered toward its cleaner vehicles when carbon matters.
3. Constraints: payload (each stop's package weight against the vehicle's
   max_payload_kg); deadlines as time windows (arrival at the stop, seconds
   after leaving the depot; SERVICE_TIME_S spent at each stop); battery (an
   electric vehicle's total expected energy within its battery).
4. A stop that cannot be served within the constraints is dropped with a
   large penalty and reported, rather than making the whole problem fail.

Baselines on the same legs, for comparison: every delivery as its own round
trip from the depot (what the one-delivery pipeline does), and a
nearest-neighbour heuristic (each vehicle in turn goes to the closest
unvisited stop that still fits).

Times are expected values; simulate_tours() drives the plan through the
Execution agent's noise model to check deadlines against simulated arrivals.
"""

import sys
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gnn"))

from schemas import DeliveryRequest, Vehicle
from vehicles import EV_REFERENCE_KWH, FleetCosts, profile

SERVICE_TIME_S = 120          # time spent handing over a parcel at each stop
DROP_PENALTY = 10_000_000     # cost of leaving a stop unserved (far above any route)
SCALE = 1000                  # OR-Tools needs integers: costs, kg and energy are multiplied by this
DEFAULT_WEIGHTS = {"time": 0.25, "delay": 0.25, "carbon": 0.25, "ev": 0.25}


def _legs(fleet: FleetCosts, vehicle_id: str, points: list, weights: dict) -> dict:
    """
    {(i, j): leg} between every pair of points for one vehicle's costs.
    A leg holds path, expected time (s), CO2 (kg, this vehicle) and energy
    (% of the 50 kWh reference battery).
    """
    solver = fleet.solver(vehicle_id)
    graph = solver.build_graph(weights)
    legs = {}
    for i, a in enumerate(points):
        # Predecessors only; paths are read back for the few points needed
        # (building every path to all 7,095 junctions was most of the time).
        pred, _ = nx.dijkstra_predecessor_and_distance(graph, a, weight="weight")
        for j, b in enumerate(points):
            if i == j:
                legs[(i, j)] = {"path": [a], "time_s": 0.0, "co2_kg": 0.0, "energy_pct": 0.0}
                continue
            if b not in pred:
                legs[(i, j)] = None
                continue
            path = [b]
            while path[-1] != a:
                path.append(pred[path[-1]][0])
            path.reverse()
            totals = solver.path_totals(path)
            legs[(i, j)] = {
                "path": path,
                "time_s": totals["expected_travel_time_s"],
                "co2_kg": totals["expected_carbon_kg"],
                "energy_pct": totals["expected_ev_energy_pct"],
            }
    return legs


def _battery_pct(vehicle: Vehicle) -> float | None:
    if not vehicle.is_ev or vehicle.battery_capacity_kwh is None:
        return None
    return vehicle.battery_capacity_kwh / EV_REFERENCE_KWH * 100.0


def _tour_totals(order: list, legs: dict, requests: list, return_to_depot: bool) -> dict:
    """Totals for one vehicle visiting stop indices `order` (1-based) from the depot (0)."""
    sequence = [0] + order + ([0] if return_to_depot and order else [])
    clock = co2 = energy = load = 0.0
    stops, path = [], [legs[(0, 0)]["path"][0]]
    for a, b in zip(sequence[:-1], sequence[1:]):
        leg = legs[(a, b)]
        clock += leg["time_s"]
        co2 += leg["co2_kg"]
        energy += leg["energy_pct"]
        path += leg["path"][1:]
        if b != 0:
            request = requests[b - 1]
            load += request.package_weight_kg or 0.0
            stops.append({
                "request_id": request.request_id, "node": request.destination, "arrival_s": clock,
                "deadline_s": request.deadline_s,
                "deadline_met": None if request.deadline_s is None else clock <= request.deadline_s,
            })
            clock += SERVICE_TIME_S
    return {"stops": stops, "path": path, "time_s": clock, "co2_kg": co2, "energy_pct": energy, "load_kg": load}


def _summary(tours: list, dropped: list) -> dict:
    stops = [s for t in tours for s in t["stops"]]
    with_deadline = [s for s in stops if s["deadline_met"] is not None]
    return {
        "total_time_s": sum(t["time_s"] for t in tours),
        "makespan_s": max((t["time_s"] for t in tours), default=0.0),
        "total_co2_kg": sum(t["co2_kg"] for t in tours),
        "stops_served": len(stops),
        "stops_dropped": len(dropped),
        "deadlines_met": sum(1 for s in with_deadline if s["deadline_met"]),
        "deadlines_total": len(with_deadline),
        "vehicles_used": sum(1 for t in tours if t["stops"]),
    }


def plan_tours(
    costs_df: pd.DataFrame, depot: int, requests: list[DeliveryRequest], vehicles: list[Vehicle],
    weights: dict = None, time_limit_s: float = 5.0, return_to_depot: bool = True,
) -> dict:
    """
    Solves the multi-stop problem. Each request's destination is a stop (its
    origin is ignored: everything leaves from `depot`); package_weight_kg and
    deadline_s are used when set. Returns the OR-Tools plan, the two
    baselines on the same legs, and the legs themselves.
    """
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    weights = weights or DEFAULT_WEIGHTS
    fleet = FleetCosts(costs_df, vehicles)
    points = [depot] + [r.destination for r in requests]
    n_points, n_vehicles = len(points), len(vehicles)

    # Legs per vehicle TYPE (vehicles of one type share them).
    legs_by_type = {}
    for v in vehicles:
        if v.vehicle_type not in legs_by_type:
            legs_by_type[v.vehicle_type] = _legs(fleet, v.vehicle_id, points, weights)
    legs_of = [legs_by_type[v.vehicle_type] for v in vehicles]
    for legs in legs_by_type.values():
        missing = [k for k, leg in legs.items() if leg is None]
        if missing:
            raise ValueError(f"No route between some points (e.g. {points[missing[0][0]]} -> {points[missing[0][1]]})")

    # Objective in real units, each term divided by its average leg value.
    off_diagonal = [leg for legs in legs_by_type.values() for (i, j), leg in legs.items() if i != j]
    time_ref = sum(leg["time_s"] for leg in off_diagonal) / len(off_diagonal) or 1.0
    co2_ref = sum(leg["co2_kg"] for leg in off_diagonal) / len(off_diagonal) or 1.0
    w_time = weights.get("time", 0) + weights.get("delay", 0)
    w_co2 = weights.get("carbon", 0) + weights.get("ev", 0)
    total_w = (w_time + w_co2) or 1.0
    w_time, w_co2 = w_time / total_w, w_co2 / total_w

    manager = pywrapcp.RoutingIndexManager(n_points, n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def counted(i, j):
        # Without a return to the depot, the final "leg" home costs nothing.
        return return_to_depot or j != 0

    def make_cost(legs):
        def cost(from_index, to_index):
            i, j = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
            if not counted(i, j):
                return 0
            leg = legs[(i, j)]
            return int(SCALE * (w_time * leg["time_s"] / time_ref + w_co2 * leg["co2_kg"] / co2_ref))
        return cost

    def make_time(legs):
        def transit(from_index, to_index):
            i, j = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
            drive_s = legs[(i, j)]["time_s"] if counted(i, j) else 0.0
            return int(drive_s + (SERVICE_TIME_S if i != 0 else 0))
        return transit

    def make_energy(legs, is_ev):
        def transit(from_index, to_index):
            i, j = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
            if not is_ev or not counted(i, j):
                return 0
            return int(SCALE * legs[(i, j)]["energy_pct"])
        return transit

    # Callbacks must outlive the solve, so keep references.
    keep = []
    time_callbacks, energy_callbacks = [], []
    for k, v in enumerate(vehicles):
        cost_fn, time_fn, energy_fn = make_cost(legs_of[k]), make_time(legs_of[k]), make_energy(legs_of[k], v.is_ev)
        keep += [cost_fn, time_fn, energy_fn]
        routing.SetArcCostEvaluatorOfVehicle(routing.RegisterTransitCallback(cost_fn), k)
        time_callbacks.append(routing.RegisterTransitCallback(time_fn))
        energy_callbacks.append(routing.RegisterTransitCallback(energy_fn))

    # Payload.
    demands = [0] + [int(round(SCALE * (r.package_weight_kg or 0.0))) for r in requests]
    demand_fn = lambda index: demands[manager.IndexToNode(index)]  # noqa: E731
    keep.append(demand_fn)
    capacities = [int(SCALE * (v.max_payload_kg if v.max_payload_kg is not None else 1e6)) for v in vehicles]
    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(demand_fn), 0, capacities, True, "Load")

    # Time and deadlines.
    horizon = int(24 * 3600)
    routing.AddDimensionWithVehicleTransits(time_callbacks, 0, horizon, True, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for idx, r in enumerate(requests, start=1):
        if r.deadline_s is not None:
            time_dim.CumulVar(manager.NodeToIndex(idx)).SetRange(0, int(r.deadline_s))

    # Battery.
    battery_caps = [int(SCALE * b) if (b := _battery_pct(v)) is not None else int(SCALE * 1e6) for v in vehicles]
    routing.AddDimensionWithVehicleTransitAndCapacity(energy_callbacks, 0, battery_caps, True, "Energy")

    for idx in range(1, n_points):
        routing.AddDisjunction([manager.NodeToIndex(idx)], DROP_PENALTY)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromMilliseconds(int(time_limit_s * 1000))
    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("OR-Tools found no solution")

    orders = []
    for k in range(n_vehicles):
        index, order = routing.Start(k), []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                order.append(node)
            index = solution.Value(routing.NextVar(index))
        orders.append(order)
    served = {i for order in orders for i in order}
    dropped = [requests[i - 1].request_id for i in range(1, n_points) if i not in served]

    def tours_for(orders_):
        tours = []
        for k, order in enumerate(orders_):
            t = _tour_totals(order, legs_of[k], requests, return_to_depot)
            p = profile(vehicles[k].vehicle_type)
            tours.append({
                "vehicle_id": vehicles[k].vehicle_id, "vehicle_type": vehicles[k].vehicle_type,
                "vehicle_label": p.label if p else "GNN reference vehicle",
                "is_ev": vehicles[k].is_ev, "battery_kwh": vehicles[k].battery_capacity_kwh,
                "max_payload_kg": vehicles[k].max_payload_kg,
                "energy_kwh": t["energy_pct"] / 100.0 * EV_REFERENCE_KWH if vehicles[k].is_ev else 0.0,
                **t,
            })
        return tours

    optimised = tours_for(orders)
    separate = _separate_trips(requests, vehicles, legs_of)
    greedy_orders, greedy_dropped = _nearest_neighbour(requests, vehicles, legs_of, return_to_depot)
    greedy = tours_for(greedy_orders)

    return {
        "points": points,
        "weights": weights,
        "objective_weights": {"time": w_time, "co2": w_co2},
        "optimised": {"tours": optimised, "dropped": dropped, **_summary(optimised, dropped)},
        "separate_trips": separate,
        "nearest_neighbour": {"tours": greedy, "dropped": greedy_dropped, **_summary(greedy, greedy_dropped)},
        "service_time_s": SERVICE_TIME_S,
        "return_to_depot": return_to_depot,
    }


def _separate_trips(requests, vehicles, legs_of) -> dict:
    """
    Every delivery as its own depot -> stop -> depot trip, on the first
    vehicle that can carry it -- the one-delivery-per-trip pipeline's way.
    Totals only (trips run one after another per vehicle, so deadlines are
    checked as if each trip started fresh: the most generous reading).
    """
    total_time = total_co2 = 0.0
    met = with_deadline = served = 0
    for idx, r in enumerate(requests, start=1):
        k = next((k for k, v in enumerate(vehicles)
                  if v.max_payload_kg is None or (r.package_weight_kg or 0) <= v.max_payload_kg), None)
        if k is None:
            continue
        out, back = legs_of[k][(0, idx)], legs_of[k][(idx, 0)]
        served += 1
        total_time += out["time_s"] + SERVICE_TIME_S + back["time_s"]
        total_co2 += out["co2_kg"] + back["co2_kg"]
        if r.deadline_s is not None:
            with_deadline += 1
            met += out["time_s"] <= r.deadline_s
    return {"total_time_s": total_time, "total_co2_kg": total_co2, "stops_served": served,
            "deadlines_met": met, "deadlines_total": with_deadline}


def _nearest_neighbour(requests, vehicles, legs_of, return_to_depot=True):
    """Each vehicle in turn: go to the closest (expected time) unvisited stop that still fits."""
    left = set(range(1, len(requests) + 1))
    orders = []
    for k, v in enumerate(vehicles):
        legs, order, here, load, energy = legs_of[k], [], 0, 0.0, 0.0
        battery = _battery_pct(v)
        while True:
            fits = []
            for i in left:
                weight = requests[i - 1].package_weight_kg or 0.0
                if v.max_payload_kg is not None and load + weight > v.max_payload_kg:
                    continue
                home = legs[(i, 0)]["energy_pct"] if return_to_depot else 0.0
                if battery is not None and energy + legs[(here, i)]["energy_pct"] + home > battery:
                    continue
                fits.append(i)
            if not fits:
                break
            nxt = min(fits, key=lambda i: legs[(here, i)]["time_s"])
            order.append(nxt)
            load += requests[nxt - 1].package_weight_kg or 0.0
            energy += legs[(here, nxt)]["energy_pct"]
            left.discard(nxt)
            here = nxt
        orders.append(order)
    return orders, [requests[i - 1].request_id for i in sorted(left)]


def simulate_tours(plan: dict, costs_df: pd.DataFrame, vehicles: list[Vehicle], seed=0, which: str = "optimised") -> list[dict]:
    """
    Drives each tour through the Execution agent's noise model (on that
    vehicle's costs) and reports simulated arrival at every stop against its
    deadline.
    """
    from feedback_loop import _edge_rows, drive  # imports the LLM client; only needed here

    fleet = FleetCosts(costs_df, vehicles)
    out = []
    for tour in plan[which]["tours"]:
        if not tour["stops"]:
            continue
        world = _edge_rows(fleet.costs(tour["vehicle_id"]))
        sim = drive(tour["path"], world, world, seed=f"{seed}|{tour['vehicle_id']}")
        # Walk the telemetry to find when each stop's node was reached, in order.
        clock, cursor, arrivals = 0.0, 0, []
        targets = [s["node"] for s in tour["stops"]]
        for t in sim["telemetry"]:
            clock += t["actual_time_s"]
            if cursor < len(targets) and t["dst"] == targets[cursor]:
                arrivals.append(clock)
                clock += SERVICE_TIME_S
                cursor += 1
        stops = []
        for s, actual in zip(tour["stops"], arrivals):
            stops.append({**s, "simulated_arrival_s": actual,
                          "simulated_deadline_met": None if s["deadline_s"] is None else actual <= s["deadline_s"]})
        out.append({"vehicle_id": tour["vehicle_id"], "stops": stops,
                    "simulated_time_s": sim["time_s"] + SERVICE_TIME_S * len(tour["stops"]),
                    "simulated_co2_kg": sim["carbon_kg"]})
    return out


# ══════════════════════════════════════════════════════════════════════════
# SMOKE TEST -- a mixed fleet and ten drop-offs on sample_costs/. No torch,
# no GNN checkpoint and no LLM key needed.
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import glob
    import random

    sys.stdout.reconfigure(encoding="utf-8")
    costs = pd.read_csv(sorted(glob.glob(str(Path(__file__).resolve().parent.parent / "sample_costs" / "*.csv")))[0])
    from routing_solver import RoutingSolver

    solver = RoutingSolver(costs)
    nodes = sorted(set(costs["src"]) | set(costs["dst"]))
    rng = random.Random(0)
    depot = 1216
    stops = []
    while len(stops) < 10:
        node = rng.choice(nodes)
        trip = solver.find_route(depot, node)
        if trip["success"] and solver.find_route(node, depot)["success"] and 600 < trip["expected_travel_time_s"] < 2400:
            stops.append(DeliveryRequest(request_id=f"stop-{len(stops)}", origin=depot, destination=node,
                                         package_weight_kg=rng.choice([20.0, 50.0, 80.0])))
    vehicles = [
        Vehicle(vehicle_id="ev-1", vehicle_type="ev_ldv", battery_capacity_kwh=30.0, current_location=depot),
        Vehicle(vehicle_id="diesel-1", vehicle_type="diesel_ldv", current_location=depot),
        Vehicle(vehicle_id="cng-3w", vehicle_type="cng_3w", max_payload_kg=300.0, current_location=depot),
    ]
    for name, w in (("fastest", {"time": 0.7, "delay": 0.2, "carbon": 0.05, "ev": 0.05}),
                    ("greenest", {"time": 0.1, "delay": 0.1, "carbon": 0.4, "ev": 0.4})):
        plan = plan_tours(costs, depot, stops, vehicles, weights=w, time_limit_s=3)
        o, s, g = plan["optimised"], plan["separate_trips"], plan["nearest_neighbour"]
        print(f"\n{name}: OR-Tools {o['total_time_s'] / 60:.0f} min driving, {o['total_co2_kg']:.2f} kg CO2, "
              f"{o['stops_served']} stops, {o['vehicles_used']} vehicles | nearest-neighbour "
              f"{g['total_time_s'] / 60:.0f} min, {g['total_co2_kg']:.2f} kg | separate trips "
              f"{s['total_time_s'] / 60:.0f} min, {s['total_co2_kg']:.2f} kg")
        for t in o["tours"]:
            print(f"   {t['vehicle_label']:34s} {len(t['stops'])} stops, {t['time_s'] / 60:5.1f} min, "
                  f"{t['co2_kg']:.2f} kg, load {t['load_kg']:.0f}/{t['max_payload_kg']:.0f} kg"
                  + (f", {t['energy_kwh']:.1f}/{t['battery_kwh']:.0f} kWh" if t["is_ev"] else ""))
