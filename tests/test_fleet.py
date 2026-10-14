"""
Tests for Phase 5 (CLAUDE.md 12): vehicle-specific carbon
(agents/vehicles.py), delivery deadlines (Coordinator, Carbon Optimizer
rule, Execution, ESG, closed loop), carbon-aware dispatch and the OR-Tools
multi-stop planner (agents/multi_stop.py).

No LLM and no torch. Uses sample_costs/ (the GNN's predictions saved as
CSV) where a real road network is needed.

    python tests/test_fleet.py        (or: pytest tests/)
"""

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn"))
sys.path.insert(0, str(ROOT / "agents"))

from coordinator_agent import CoordinatorAgent  # noqa: E402
from dispatcher_agent import DispatcherAgent  # noqa: E402
from esg_reporter_agent import ESGReporterAgent  # noqa: E402
from execution_agent import ExecutionAgent  # noqa: E402
from routing_solver import RoutingSolver  # noqa: E402
from route_agent import build_route_result  # noqa: E402
from schemas import DeliveryRequest, Vehicle  # noqa: E402
from vehicles import (GRID_KG_CO2_PER_KWH, REFERENCE_KG_CO2_PER_KM, VEHICLE_TYPES, FleetCosts,  # noqa: E402
                      costs_for_vehicle, vehicle_co2_kg, vehicle_energy_kwh)

GREEN = {"time": 0.1, "delay": 0.1, "carbon": 0.4, "ev": 0.4}
TIME_ONLY = {"time": 1.0, "delay": 0.0, "carbon": 0.0, "ev": 0.0}
_COSTS = None


def costs() -> pd.DataFrame:
    global _COSTS
    if _COSTS is None:
        _COSTS = pd.read_csv(sorted((ROOT / "sample_costs").glob("*.csv"))[0])
    return _COSTS


def trip(solver, rng_nodes, min_s=900, max_s=3000):
    for a, b in rng_nodes:
        r = solver.find_route(a, b)
        if r["success"] and min_s < r["expected_travel_time_s"] < max_s:
            return a, b
    raise AssertionError("no suitable trip found")


def node_pairs(seed=0):
    import random
    nodes = sorted(set(costs()["src"]) | set(costs()["dst"]))
    rng = random.Random(seed)
    while True:
        yield tuple(rng.sample(nodes, 2))


def test_catalogue_and_conversions():
    assert set(VEHICLE_TYPES) == {"diesel_ldv", "diesel_mdv", "diesel_3w", "cng_3w", "petrol_2w", "ev_ldv"}
    assert abs(VEHICLE_TYPES["diesel_ldv"].carbon_scale - 0.3070 / REFERENCE_KG_CO2_PER_KM) < 1e-12
    assert vehicle_co2_kg(None, 1.23, 4.0) == 1.23                       # reference vehicle unchanged
    assert abs(vehicle_co2_kg("diesel_mdv", 1.0, 0.0) - 0.5928 / 0.27) < 1e-12
    # EV: 10% of the 50 kWh battery = 5 kWh -> 5 x 0.727 kg
    assert abs(vehicle_co2_kg("ev_ldv", 99.0, 10.0) - 5 * GRID_KG_CO2_PER_KWH) < 1e-12
    assert vehicle_energy_kwh("diesel_ldv", 10.0) == 0.0 and vehicle_energy_kwh("ev_ldv", 10.0) == 5.0
    try:
        vehicle_co2_kg("hovercraft", 1.0, 1.0)
        raise AssertionError("unknown type accepted")
    except ValueError:
        pass


def test_vehicle_schema_follows_its_type():
    ev = Vehicle(vehicle_id="a", vehicle_type="ev_ldv", battery_capacity_kwh=30)
    diesel = Vehicle(vehicle_id="b", vehicle_type="diesel_ldv", is_ev=True)   # type wins
    scooter = Vehicle(vehicle_id="c", vehicle_type="petrol_2w", max_payload_kg=15)
    untyped = Vehicle(vehicle_id="d")
    assert ev.is_ev and ev.max_payload_kg == 750.0
    assert not diesel.is_ev
    assert scooter.max_payload_kg == 15                                        # explicit value kept
    assert untyped.vehicle_type is None and not untyped.is_ev and untyped.max_payload_kg is None


def test_costs_for_vehicle_and_routes():
    base = costs()
    assert costs_for_vehicle(base, None) is base
    diesel = costs_for_vehicle(base, "diesel_ldv")
    ev = costs_for_vehicle(base, "ev_ldv")
    assert (diesel["travel_time_s"] == base["travel_time_s"]).all()
    ratio = diesel["carbon_kg"] / base["carbon_kg"].where(base["carbon_kg"] > 0)
    assert abs(ratio.dropna() - 0.3070 / 0.27).max() < 1e-9
    assert abs(ev["carbon_kg"] - ev["ev_energy_pct"] / 100 * 50 * GRID_KG_CO2_PER_KWH).max() < 1e-12
    # A combustion type never changes the route (uniform scale); an EV may.
    ref, dsl = RoutingSolver(base), RoutingSolver(diesel)
    pairs = node_pairs(1)
    for _ in range(20):
        a, b = trip(ref, pairs)
        assert ref.find_route(a, b, weights=GREEN)["path"] == dsl.find_route(a, b, weights=GREEN)["path"]


def test_fleet_costs_routes_each_vehicle_on_its_own_costs():
    vehicles = [Vehicle(vehicle_id="ev", vehicle_type="ev_ldv"), Vehicle(vehicle_id="plain")]
    fleet = FleetCosts(costs(), vehicles)
    assert fleet.costs("plain") is costs() and fleet.costs("nobody") is costs()
    assert fleet.solver("ev") is fleet.solver("ev")                        # built once
    a, b = trip(fleet.solver("plain"), node_pairs(2))
    r = fleet.solver("ev").find_route(a, b, weights=GREEN)
    assert abs(r["total_carbon_kg"] - r["total_ev_energy_pct"] / 100 * 50 * GRID_KG_CO2_PER_KWH) < 1e-9


def test_coordinator_meets_a_deadline_with_the_smallest_change():
    base = costs()
    solver = RoutingSolver(base)
    a, b = trip(solver, node_pairs(3), 1800, 4000)
    green = solver.find_route(a, b, weights=GREEN)
    fast = solver.find_route(a, b, weights=TIME_ONLY)
    if green["expected_travel_time_s"] - fast["expected_travel_time_s"] < 60:
        return   # this trip has no time to trade; nothing to test here
    deadline = (green["expected_travel_time_s"] + fast["expected_travel_time_s"]) / 2
    request = DeliveryRequest(request_id="r", origin=a, destination=b, deadline_s=deadline, vehicle_id="v")
    result = build_route_result(request, green, GREEN, None)
    assert result.deadline_s == deadline
    final, rounds, ok = CoordinatorAgent(base, [Vehicle(vehicle_id="v")], edge_capacity=5).resolve([result], {"r": request})
    assert ok and rounds == 1
    assert final[0].expected_travel_time_s <= deadline
    assert "Deadline" in final[0].reasoning and final[0].weights_used != TIME_ONLY or deadline < fast["expected_travel_time_s"] * 1.01

    # An impossible deadline: fastest route taken, plan not fully resolved.
    impossible = DeliveryRequest(request_id="r", origin=a, destination=b, deadline_s=fast["expected_travel_time_s"] / 2,
                                 vehicle_id="v")
    result = build_route_result(impossible, green, GREEN, None)
    final, _, ok = CoordinatorAgent(base, [Vehicle(vehicle_id="v")], edge_capacity=5, max_rounds=2).resolve(
        [result], {"r": impossible})
    assert not ok and "misses it" in final[0].reasoning
    assert abs(final[0].expected_travel_time_s - fast["expected_travel_time_s"]) < 1e-6


def test_no_deadline_changes_nothing_in_the_coordinator():
    base = costs()
    solver = RoutingSolver(base)
    a, b = trip(solver, node_pairs(4))
    request = DeliveryRequest(request_id="r", origin=a, destination=b, vehicle_id="v")
    result = build_route_result(request, solver.find_route(a, b, weights=GREEN), GREEN, "kept")
    final, rounds, ok = CoordinatorAgent(base, [Vehicle(vehicle_id="v")], edge_capacity=5).resolve([result], {"r": request})
    assert ok and rounds == 0 and final[0].path == result.path and final[0].reasoning == "kept"


def test_execution_and_esg_report_deadlines_and_vehicle_types():
    base = costs()
    solver = RoutingSolver(base)
    a, b = trip(solver, node_pairs(5))
    vehicles = [Vehicle(vehicle_id="d", vehicle_type="diesel_ldv"), Vehicle(vehicle_id="e", vehicle_type="ev_ldv",
                                                                            battery_capacity_kwh=60)]
    fleet = FleetCosts(base, vehicles)
    results = []
    for vid, deadline in (("d", 10_000.0), ("e", None)):
        req = DeliveryRequest(request_id=vid, origin=a, destination=b, deadline_s=deadline, vehicle_id=vid)
        r = build_route_result(req, fleet.solver(vid).find_route(a, b, weights=GREEN), GREEN, None)
        r.vehicle_type = fleet.vehicle_type(vid)
        results.append(r)
    reports = ExecutionAgent(base, seed=0).run(results, {v.vehicle_id: v for v in vehicles})
    by_id = {rep["request_id"]: rep for rep in reports}
    assert by_id["d"]["deadline_met"] is True and by_id["e"]["deadline_met"] is None
    # Simulated carbon is in each vehicle's own terms.
    ratio = by_id["d"]["actual_carbon_kg"] / by_id["d"]["actual_time_s"]
    assert by_id["d"]["vehicle_type"] == "diesel_ldv" and ratio > 0
    metrics = ESGReporterAgent().aggregate(results, [], {v.vehicle_id: v for v in vehicles}, [], reports, 0)
    assert metrics["num_with_deadline"] == 1 and metrics["deadline_met_rate"] == 1.0
    assert metrics["vehicle_types"] == ["diesel_ldv", "ev_ldv"]


def test_dispatcher_untyped_unchanged_and_typed_prefers_cleaner():
    base = costs()
    solver = RoutingSolver(base)
    a, b = trip(solver, node_pairs(6))
    near = next(n for n in solver.find_route(a, b)["path"][:1])   # the pickup itself
    # Two vehicles at the pickup: identical distance, so only carbon can separate them.
    req = lambda: DeliveryRequest(request_id="r", origin=a, destination=b, package_weight_kg=50)  # noqa: E731
    plain = [Vehicle(vehicle_id="x", current_location=near), Vehicle(vehicle_id="y", current_location=near)]
    assigned, _ = DispatcherAgent(base).assign([req()], plain)
    assert assigned[0].vehicle_id == "x"                                  # tie -> first, as before
    typed = [Vehicle(vehicle_id="diesel", vehicle_type="diesel_mdv", current_location=near),
             Vehicle(vehicle_id="cng", vehicle_type="cng_3w", current_location=near)]
    dispatcher = DispatcherAgent(base)
    assigned, _ = dispatcher.assign([req()], typed)
    assert assigned[0].vehicle_id == "cng"
    assert dispatcher.last_assignments[0]["trip_co2_kg"] > 0


def test_multi_stop_plan_is_feasible_and_beats_the_baselines():
    from multi_stop import plan_tours, simulate_tours

    base = costs()
    solver = RoutingSolver(base)
    depot = 1216
    pairs = node_pairs(7)
    stops = []
    while len(stops) < 7:
        a, node = next(pairs)
        out, back = solver.find_route(depot, node), solver.find_route(node, depot)
        if out["success"] and back["success"] and 600 < out["expected_travel_time_s"] < 2400:
            stops.append(DeliveryRequest(request_id=f"s{len(stops)}", origin=depot, destination=node,
                                         package_weight_kg=40.0))
    vehicles = [Vehicle(vehicle_id="ev", vehicle_type="ev_ldv", battery_capacity_kwh=40, max_payload_kg=160),
                Vehicle(vehicle_id="diesel", vehicle_type="diesel_ldv", max_payload_kg=160)]
    plan = plan_tours(base, depot, stops, vehicles, weights=GREEN, time_limit_s=2)
    o = plan["optimised"]
    assert o["stops_served"] == 7 and not o["dropped"]
    seen = [s["request_id"] for t in o["tours"] for s in t["stops"]]
    assert sorted(seen) == sorted(r.request_id for r in stops)          # every stop exactly once
    for t in o["tours"]:
        assert t["load_kg"] <= t["max_payload_kg"] + 1e-9
        if t["is_ev"]:
            assert t["energy_kwh"] <= t["battery_kwh"] + 1e-6
        if t["stops"]:
            assert t["path"][0] == depot and t["path"][-1] == depot
            arrivals = [s["arrival_s"] for s in t["stops"]]
            assert arrivals == sorted(arrivals)
    cost = lambda r: 0.5 * r["total_time_s"] / 3600 + 0.5 * r["total_co2_kg"]  # noqa: E731
    assert o["total_co2_kg"] < plan["separate_trips"]["total_co2_kg"]
    assert o["total_co2_kg"] <= plan["nearest_neighbour"]["total_co2_kg"] + 1e-9 or cost(o) <= cost(plan["nearest_neighbour"])
    sims = simulate_tours(plan, base, vehicles, seed=0)
    assert sum(len(s["stops"]) for s in sims) == 7


def test_multi_stop_deadlines_and_dropping():
    from multi_stop import plan_tours

    base = costs()
    solver = RoutingSolver(base)
    depot = 1216
    pairs = node_pairs(8)
    stops = []
    while len(stops) < 4:
        _, node = next(pairs)
        out, back = solver.find_route(depot, node), solver.find_route(node, depot)
        if out["success"] and back["success"] and 900 < out["expected_travel_time_s"] < 2400:
            stops.append(DeliveryRequest(request_id=f"s{len(stops)}", origin=depot, destination=node,
                                         package_weight_kg=10.0))
    direct = solver.find_route(depot, stops[0].destination)["expected_travel_time_s"]
    stops[0].deadline_s = direct * 1.1                    # must be the first stop of some vehicle
    stops[1].deadline_s = 60.0                            # impossible: dropped, not a failure
    vehicles = [Vehicle(vehicle_id="a"), Vehicle(vehicle_id="b")]
    plan = plan_tours(base, depot, stops, vehicles, weights=TIME_ONLY, time_limit_s=2)
    o = plan["optimised"]
    assert o["dropped"] == ["s1"]
    first_stops = [t["stops"][0]["request_id"] for t in o["tours"] if t["stops"]]
    assert "s0" in first_stops
    met = [s for t in o["tours"] for s in t["stops"] if s["deadline_s"] is not None]
    assert all(s["deadline_met"] for s in met)

    # Open tours: no return leg is counted.
    open_plan = plan_tours(base, depot, stops[2:], vehicles[:1], weights=TIME_ONLY, time_limit_s=1, return_to_depot=False)
    tour = open_plan["optimised"]["tours"][0]
    assert tour["path"][-1] == tour["stops"][-1]["node"]


def test_closed_loop_deadline_rule():
    from disruptions import apply_incident
    from feedback_loop import ClosedLoopExecutor

    sys.path.insert(0, str(ROOT / "tests"))
    from test_feedback_loop import TIME_ONLY as T, ladder, main_road

    road = ladder()
    after = apply_incident(road, {(8, 9), (9, 10)})
    planner = RoutingSolver(road)
    promised = planner.path_totals(main_road())["expected_travel_time_s"]
    # Generous deadline: both options meet it, so the deadline rule stays out of the way.
    rep = ClosedLoopExecutor(road, after, seed=0, use_llm=False).run(main_road(), T, deadline_s=promised * 3)
    assert rep["decision"]["rule"] == "policy"
    # A deadline between the two options' expected arrivals: only the detour meets it.
    elapsed = rep["position"]["elapsed_s"]
    stay_at = elapsed + rep["stay"]["expected_time_s"]
    detour_at = elapsed + rep["alternative"]["expected_time_s"]
    assert detour_at < stay_at
    rep = ClosedLoopExecutor(road, after, seed=0, use_llm=False).run(
        main_road(), T, deadline_s=(stay_at + detour_at) / 2)
    assert rep["decision"]["rule"] == "deadline" and rep["decision"]["replan"]
    # Neither meets it: the rule steps aside and the usual comparison decides.
    rep = ClosedLoopExecutor(road, after, seed=0, use_llm=False).run(main_road(), T, deadline_s=detour_at / 2)
    assert rep["decision"]["rule"] == "policy"


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        started = time.perf_counter()
        try:
            fn()
            print(f"PASS  {name}  ({time.perf_counter() - started:.1f}s)")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}  {e!r}")
    print(f"\n{len(tests) - failed} of {len(tests)} passed")
    sys.exit(1 if failed else 0)
