"""
Tests for the closed loop (agents/feedback_loop.py), the disruption models
(agents/disruptions.py) and the solver pieces they rely on
(RoutingSolver.path_totals and reference_costs).

No LLM, no torch and no GNN checkpoint: everything runs on a small
hand-built road network, plus one check on sample_costs/ if it is there.

    python tests/test_feedback_loop.py        (or: pytest tests/)
"""

import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn"))
sys.path.insert(0, str(ROOT / "agents"))

from disruptions import apply_incident, edge_at_fraction, edges_near  # noqa: E402
from feedback_loop import ClosedLoopExecutor, FeedbackStore, drive, fleet_learning  # noqa: E402
from routing_solver import RoutingSolver  # noqa: E402
from schemas import Vehicle  # noqa: E402

TIME_ONLY = {"time": 1.0, "delay": 0.0, "carbon": 0.0, "ev": 0.0}


def ladder(rungs: int = 12) -> pd.DataFrame:
    """
    Two parallel roads from node 0 to the last node, joined at every step:

        0 - 1 - 2 - ... - n        the main road, 60 s a segment
        |   |   |         |        links, 30 s
        A - B - C - ... - Z        the side road, 80 s a segment

    So the plan is the main road, and the side road is always available as a
    detour around any one stretch of it. Varied carbon/EV so that the
    solver's percentile scaling has a range to work with.
    """
    rows = []

    def road(u, v, seconds, carbon):
        for a, b in ((u, v), (v, u)):
            rows.append({"src": a, "dst": b, "travel_time_s": float(seconds), "delay_probability": 0.2,
                         "carbon_kg": carbon, "ev_energy_pct": carbon * 2.0})

    for i in range(rungs):
        road(i, i + 1, 60, 0.10)
        road(100 + i, 100 + i + 1, 80, 0.14)
    for i in range(rungs + 1):
        road(i, 100 + i, 30, 0.05)
    return pd.DataFrame(rows)


def main_road(rungs: int = 12) -> list:
    return list(range(rungs + 1))


def test_path_totals_matches_find_route():
    solver = RoutingSolver(ladder())
    found = solver.find_route(0, 12, weights=TIME_ONLY)
    totals = solver.path_totals(found["path"])
    for key, value in totals.items():
        assert found[key] == value, key
    assert found["path"] == main_road()


def test_reference_costs_change_nothing_when_costs_are_unchanged():
    costs = ladder()
    plain = RoutingSolver(costs)
    referenced = RoutingSolver(costs, reference_costs=costs)
    for weights in (TIME_ONLY, {"time": 0.25, "delay": 0.25, "carbon": 0.25, "ev": 0.25}):
        path = plain.find_route(0, 12, weights=weights)["path"]
        assert referenced.find_route(0, 12, weights=weights)["path"] == path
        assert abs(plain.route_cost(path, weights) - referenced.route_cost(path, weights)) < 1e-12


def test_reference_costs_do_not_cap_a_slowed_road():
    # A road that is already at the top of the range (norm 1.0) and then
    # becomes 4x slower must cost 4x more, not the same as before.
    costs = ladder()
    slowed = apply_incident(costs, {(5, 6)}, slowdown=4.0, delay_probability=0.2)
    before = RoutingSolver(costs).route_cost([5, 6], TIME_ONLY)
    rescaled = RoutingSolver(slowed).route_cost([5, 6], TIME_ONLY)
    referenced = RoutingSolver(slowed, reference_costs=costs).route_cost([5, 6], TIME_ONLY)
    assert rescaled <= 1.0 + 1e-9                      # the plain solver caps it
    assert abs(referenced - 4.0 * before) < 1e-9       # the referenced one does not


def test_edges_near_and_apply_incident():
    costs = ladder()
    assert edges_near(costs, (5, 6), hops=0) == {(5, 6), (6, 5)}
    ball = edges_near(costs, (5, 6), hops=1)
    assert {(4, 5), (6, 7), (5, 105), (106, 6)} <= ball
    assert (105, 106) in ball and (3, 4) not in ball

    after = apply_incident(costs, {(5, 6)})
    hit = (after["src"] == 5) & (after["dst"] == 6)
    assert (after.loc[hit, "travel_time_s"] == 240.0).all()
    assert (after.loc[hit, "delay_probability"] == 0.9).all()
    assert after.loc[~hit].equals(costs.loc[~hit])
    assert costs.loc[hit, "travel_time_s"].iloc[0] == 60.0   # the input is not modified


def test_edge_at_fraction():
    path = main_road()
    times = {(i, i + 1): 60.0 for i in range(12)}
    assert edge_at_fraction(path, times, 0.0) == (0, 1)
    assert edge_at_fraction(path, times, 0.5) == (5, 6)
    assert edge_at_fraction(path, times, 1.0) == (11, 12)


def test_nothing_changed_means_no_replan():
    costs = ladder()
    report = ClosedLoopExecutor(costs, costs, seed=1, use_llm=False).run(main_road(), TIME_ONLY)
    assert report["disruption_reached"]
    assert report["decision"]["rule"] == "same_route" and not report["decision"]["replan"]
    assert report["final_path"] == main_road()
    assert report["counterfactual"] is None
    assert abs(report["slowdown_ahead"] - 1.0) < 1e-12


def test_incident_ahead_is_avoided_and_saves_time():
    costs = ladder()
    after = apply_incident(costs, {(8, 9), (9, 10)})
    saved = []
    for seed in range(40):
        report = ClosedLoopExecutor(costs, after, seed=seed, use_llm=False).run(
            main_road(), TIME_ONLY, disruption_after_fraction=0.3)
        assert report["decision"]["replan"] and report["decision"]["rule"] == "policy"
        path = report["final_path"]
        assert path[0] == 0 and path[-1] == 12
        assert (8, 9) not in set(zip(path[:-1], path[1:]))
        done = report["position"]["edges_done"]
        assert path[:done + 1] == main_road()[:done + 1]      # the part already driven is kept
        assert len(report["telemetry"]) == len(path) - 1
        saved.append(report["saved"]["time_s"])
    assert sum(saved) / len(saved) > 120       # two 4x segments are worth minutes
    assert min(saved) > 0


def test_an_incident_already_passed_changes_nothing():
    costs = ladder()
    after = apply_incident(costs, {(0, 1), (1, 2)})
    report = ClosedLoopExecutor(costs, after, seed=0, use_llm=False).run(
        main_road(), TIME_ONLY, disruption_after_fraction=0.5)
    assert report["decision"]["rule"] == "same_route"
    assert abs(report["slowdown_ahead"] - 1.0) < 1e-12


def test_not_informed_keeps_the_plan_but_reports_what_it_cost():
    costs = ladder()
    after = apply_incident(costs, {(8, 9), (9, 10)})
    told = ClosedLoopExecutor(costs, after, seed=3, use_llm=False, informed=True).run(main_road(), TIME_ONLY)
    untold = ClosedLoopExecutor(costs, after, seed=3, use_llm=False, informed=False).run(main_road(), TIME_ONLY)
    assert untold["decision"]["rule"] == "not_informed" and untold["final_path"] == main_road()
    # Same seed, same luck: one run's outcome is the other's counterfactual.
    assert untold["outcome"] == told["counterfactual"]
    assert untold["counterfactual"] == told["outcome"]
    # Its telemetry is measured against what the planner believed.
    slow = [t for t in untold["telemetry"] if (t["src"], t["dst"]) in {(8, 9), (9, 10)}]
    assert len(slow) == 2 and all(t["actual_time_s"] / t["expected_time_s"] > 2.5 for t in slow)


def test_arrived_before_the_change():
    costs = ladder()
    after = apply_incident(costs, {(8, 9)})
    report = ClosedLoopExecutor(costs, after, seed=0, use_llm=False).run(
        main_road(), TIME_ONLY, disruption_after_fraction=5.0)
    assert not report["disruption_reached"] and report["decision"]["rule"] == "arrived"
    assert report["final_path"] == main_road()


def test_ev_hard_rule_forces_a_switch_and_blocks_an_unaffordable_one():
    costs = ladder()
    after = apply_incident(costs, {(8, 9), (9, 10)})   # 4x energy on two segments of the plan
    planner = RoutingSolver(costs)
    needed = planner.path_totals(main_road())["expected_ev_energy_pct"]

    # Enough battery for the plan as promised (plus 15%), not for the plan with the incident.
    vehicle = Vehicle(vehicle_id="ev", is_ev=True, battery_capacity_kwh=needed * 1.15 / 100 * 50)
    report = ClosedLoopExecutor(costs, after, seed=0, use_llm=False, planner=planner).run(
        main_road(), TIME_ONLY, vehicle=vehicle, disruption_after_fraction=0.3)
    assert report["decision"]["rule"] == "ev_range" and report["decision"]["replan"]
    assert report["alternative"]["expected_ev_energy_pct"] <= report["battery_left_pct"]
    assert report["stay"]["expected_ev_energy_pct"] > report["battery_left_pct"]

    # A battery too small for either: stay unless the alternative needs less.
    tiny = Vehicle(vehicle_id="ev", is_ev=True, battery_capacity_kwh=needed * 0.5 / 100 * 50)
    report = ClosedLoopExecutor(costs, after, seed=0, use_llm=False, planner=planner).run(
        main_road(), TIME_ONLY, vehicle=tiny, disruption_after_fraction=0.3)
    assert report["decision"]["rule"] == "ev_range"
    assert report["decision"]["replan"] == (
        report["alternative"]["expected_ev_energy_pct"] < report["stay"]["expected_ev_energy_pct"])


def test_small_gain_is_not_worth_switching():
    # A 5% slowdown on one segment ahead: the side road is not better, and a
    # marginally better route would be under the threshold anyway.
    costs = ladder()
    after = apply_incident(costs, {(8, 9)}, slowdown=1.05, delay_probability=0.2)
    report = ClosedLoopExecutor(costs, after, seed=0, use_llm=False).run(main_road(), TIME_ONLY)
    assert not report["decision"]["replan"]
    assert report["decision"]["rule"] in ("same_route", "small_gain", "dominated")


def test_drive_is_paired_and_reproducible():
    world = {(r.src, r.dst): r for r in ladder().itertuples(index=False)}
    a = drive(main_road(), world, world, seed=7)
    b = drive(main_road(), world, world, seed=7)
    assert a == b
    side = [0, 1, 2, 102, 103, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    c = drive(side, world, world, seed=7)
    shared = {(t["src"], t["dst"]): t["actual_time_s"] for t in a["telemetry"]}
    for t in c["telemetry"]:
        if (t["src"], t["dst"]) in shared:
            assert t["actual_time_s"] == shared[(t["src"], t["dst"])]
    assert drive(main_road(), world, world, seed=8)["time_s"] != a["time_s"]


def test_feedback_store_ignores_ordinary_luck_and_learns_an_incident():
    store = FeedbackStore()
    assert abs(store.deadband - 0.4) < 1e-9
    # The worst single report a healthy road can give (ratio 2.2) is not enough.
    store.record([{"src": 1, "dst": 2, "expected_time_s": 100.0, "actual_time_s": 220.0}])
    assert store.factor((1, 2)) == 1.0 and store.corrections() == {}
    # A 4x incident clears the deadband at once and settles as reports come in.
    factors = []
    for _ in range(6):
        store.record([{"src": 5, "dst": 6, "expected_time_s": 100.0, "actual_time_s": 400.0}])
        factors.append(store.factor((5, 6)))
    assert factors[0] > 1.4 and factors == sorted(factors) and 3.0 < factors[-1] < 4.0
    assert store.factor((9, 9)) == 1.0     # never observed

    costs = ladder()
    corrected = store.corrected(costs)
    hit = (costs["src"] == 5) & (costs["dst"] == 6)
    assert abs(corrected.loc[hit, "travel_time_s"].iloc[0] - 60.0 * factors[-1]) < 1e-9
    assert corrected.loc[~hit].equals(costs.loc[~hit])
    assert (corrected["delay_probability"] == costs["delay_probability"]).all()

    # The GNN returns float32 columns; correcting them must not raise.
    as_float32 = costs.astype({c: "float32" for c in ("travel_time_s", "delay_probability", "carbon_kg", "ev_energy_pct")})
    corrected32 = store.corrected(as_float32)
    assert abs(corrected32.loc[hit, "travel_time_s"].iloc[0] - 60.0 * factors[-1]) < 1e-4
    slowed32 = apply_incident(as_float32, {(5, 6)})
    assert abs(slowed32.loc[hit, "travel_time_s"].iloc[0] - 240.0) < 1e-4

    with tempfile.TemporaryDirectory() as folder:
        store.save(Path(folder) / "store.json")
        again = FeedbackStore.load(Path(folder) / "store.json")
    assert again.corrections() == store.corrections()


def test_fleet_learning_finds_the_informed_route():
    costs = ladder()
    after = apply_incident(costs, {(8, 9), (9, 10)})
    learned = fleet_learning(costs, after, 0, 12, TIME_ONLY, n_vehicles=6, seed=0)
    trips = learned["trips"]
    assert trips[0]["same_as_no_feedback"] and trips[0]["segments_corrected_before_planning"] == 0
    assert all(t["same_as_informed"] for t in trips[1:])      # one report was enough here
    assert all(t["actual_time_s"] < t["no_feedback_time_s"] for t in trips[1:])
    # Nothing to learn in an unchanged world: every vehicle keeps the plan.
    calm = fleet_learning(costs, costs, 0, 12, TIME_ONLY, n_vehicles=6, seed=0)
    assert all(t["same_as_no_feedback"] for t in calm["trips"])


def test_on_the_bengaluru_sample():
    files = sorted((ROOT / "sample_costs").glob("*.csv"))
    if not files:
        print("      (skipped: no sample_costs/*.csv)")
        return
    costs = pd.read_csv(files[0])
    weights = {"time": 0.7, "delay": 0.2, "carbon": 0.05, "ev": 0.05}
    planner = RoutingSolver(costs)
    nodes = pd.unique(costs[["src", "dst"]].values.ravel())
    path = planner.find_route(int(nodes[100]), int(nodes[6000]), weights=weights)["path"]
    edge_time = dict(zip(zip(costs["src"], costs["dst"]), costs["travel_time_s"]))
    blocked = edges_near(costs, edge_at_fraction(path, edge_time, 0.7), hops=1)
    after = apply_incident(costs, blocked)

    report = ClosedLoopExecutor(costs, after, seed=0, use_llm=False, planner=planner).run(path, weights)
    assert report["decision"]["replan"] and report["saved"]["time_s"] > 0
    assert report["final_path"][0] == path[0] and report["final_path"][-1] == path[-1]
    calm = ClosedLoopExecutor(costs, costs, seed=0, use_llm=False, planner=planner).run(path, weights)
    assert calm["decision"]["rule"] == "same_route"


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
