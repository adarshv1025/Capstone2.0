"""
multi_stop_experiment.py

The OR-Tools multi-stop planner (agents/multi_stop.py) against its two
baselines on random delivery rounds, for two fleets:
    diesel  three diesel light goods vehicles
    mixed   an electric light goods vehicle, a diesel one, a CNG three-wheeler
Same depot, stops, parcels and deadlines for both fleets, so the difference
between them is what the choice of vehicle is worth.

Each instance: a random depot, STOPS random drop-offs 10-40 minutes away,
parcels of 10-80 kg, vehicles carrying 300 kg each (so at least two are
needed), and a deadline on every other stop (1.5 x its direct time + 20
minutes). Priorities fastest and greenest. Runs on sample_costs/ (the GNN's
predictions for one snapshot), so it needs no torch and no LLM.

    python agents/multi_stop_experiment.py                 # 20 instances, about 4 minutes
    python agents/multi_stop_experiment.py --instances 5   # quick look

Writes agents/evaluation_results/multi_stop_experiment.csv.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agents"))
sys.path.insert(0, str(ROOT / "gnn"))

from multi_stop import plan_tours, simulate_tours  # noqa: E402
from routing_solver import RoutingSolver  # noqa: E402
from schemas import DeliveryRequest, Vehicle  # noqa: E402

OUT = ROOT / "agents" / "evaluation_results" / "multi_stop_experiment.csv"
STOPS = 10
PRIORITIES = {
    "fastest": {"time": 0.7, "delay": 0.2, "carbon": 0.05, "ev": 0.05},
    "greenest": {"time": 0.1, "delay": 0.1, "carbon": 0.4, "ev": 0.4},
}


def fleets(depot):
    return {
        "diesel": [Vehicle(vehicle_id=f"diesel-{i}", vehicle_type="diesel_ldv", max_payload_kg=300.0,
                           current_location=depot) for i in range(3)],
        "mixed": [Vehicle(vehicle_id="ev", vehicle_type="ev_ldv", battery_capacity_kwh=30.0, max_payload_kg=300.0,
                          current_location=depot),
                  Vehicle(vehicle_id="diesel", vehicle_type="diesel_ldv", max_payload_kg=300.0, current_location=depot),
                  Vehicle(vehicle_id="cng-3w", vehicle_type="cng_3w", max_payload_kg=300.0, current_location=depot)],
    }


def make_instance(solver, nodes, rng):
    while True:
        depot = rng.choice(nodes)
        stops, tries = [], 0
        while len(stops) < STOPS and tries < 400:
            tries += 1
            node = rng.choice(nodes)
            out, back = solver.find_route(depot, node), solver.find_route(node, depot)
            if not (out["success"] and back["success"]) or not 600 <= out["expected_travel_time_s"] <= 2400:
                continue
            deadline = 1.5 * out["expected_travel_time_s"] + 1200 if len(stops) % 2 == 0 else None
            stops.append(DeliveryRequest(request_id=f"s{len(stops)}", origin=depot, destination=node,
                                         package_weight_kg=float(rng.choice([10, 20, 40, 60, 80])), deadline_s=deadline))
        if len(stops) == STOPS:
            return depot, stops


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=2.0, help="OR-Tools seconds per solve")
    args = parser.parse_args()

    costs = pd.read_csv(sorted((ROOT / "sample_costs").glob("*.csv"))[0])
    solver = RoutingSolver(costs)
    nodes = sorted(set(costs["src"]) | set(costs["dst"]))
    rng = random.Random(args.seed)
    rows = []
    started = time.perf_counter()
    for i in range(args.instances):
        depot, stops = make_instance(solver, nodes, rng)
        for fleet_name, vehicles in fleets(depot).items():
            for priority, weights in PRIORITIES.items():
                plan = plan_tours(costs, depot, stops, vehicles, weights=weights, time_limit_s=args.time_limit)
                sims = simulate_tours(plan, costs, vehicles, seed=f"{args.seed}|{i}|{fleet_name}|{priority}")
                sim_stops = [s for t in sims for s in t["stops"] if s["deadline_s"] is not None]
                for method in ("optimised", "nearest_neighbour", "separate_trips"):
                    r = plan[method]
                    rows.append({
                        "instance": i, "fleet": fleet_name, "priority": priority, "method": method,
                        "total_time_min": r["total_time_s"] / 60, "total_co2_kg": r["total_co2_kg"],
                        "stops_served": r["stops_served"], "deadlines_met": r["deadlines_met"],
                        "deadlines_total": r["deadlines_total"],
                        "vehicles_used": r.get("vehicles_used"),
                        "co2_by_vehicle_type": None if method == "separate_trips" else str(
                            {t["vehicle_type"]: round(t["co2_kg"], 3) for t in r["tours"] if t["stops"]}),
                        "simulated_deadlines_met": (sum(1 for s in sim_stops if s["simulated_deadline_met"])
                                                    if method == "optimised" else None),
                    })
        print(f"  instance {i + 1}/{args.instances} done ({time.perf_counter() - started:.0f}s)")

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    pd.set_option("display.width", 220, "display.max_columns", 20)
    summary = df.groupby(["fleet", "priority", "method"], sort=False).agg(
        time_min=("total_time_min", "mean"), co2_kg=("total_co2_kg", "mean"),
        served=("stops_served", "mean"), deadlines_met=("deadlines_met", "sum"),
        deadlines_total=("deadlines_total", "sum"), vehicles_used=("vehicles_used", "mean"),
        # Only the OR-Tools plan is simulated; blank (not 0) for the others.
        simulated_deadlines_met=("simulated_deadlines_met", lambda s: s.sum(min_count=1)),
    ).round(2)
    print(f"\n{args.instances} instances x {STOPS} stops, means per instance:")
    print(summary.to_string())
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
