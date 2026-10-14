"""
replan_experiment.py

Does closing the loop help? Runs many trips through agents/feedback_loop.py
under each kind of mid-trip change and compares, trip by trip and with the
same luck, finishing the planned route against re-planning.

Kinds of change (see agents/disruptions.py and gnn/snapshot_disruptions.py
for what each one can claim):
    none      nothing changes (control: the loop must leave the plan alone)
    incident  WORLD-level: a 4x slowdown around a point 65% of the way along
              the route -- an assumption, not a GNN prediction
    rain      GNN-level: rain on every road, scored by the GNN
    surge     GNN-level: congestion raised to 1.0 on the route from 45% of
              the way on, scored by the GNN
    update    GNN-level: the next snapshot of the same day arrives

The change happens 30% of the way into each trip's promised time. The
re-plan decision is the deterministic policy (no LLM), so a run costs no
tokens and is exactly repeatable. For the incident it also runs the long
loop (fleet_learning): the incident is never reported and five vehicles in
a row have to find it from telemetry.

Needs the GNN checkpoint and dataset_kaggle/test (see README).

    python agents/replan_experiment.py                  # 120 trips, about 3 minutes
    python agents/replan_experiment.py --trips 20       # quick look

Writes agents/evaluation_results/replan_experiment.csv (one row per trip
and kind), replan_summary.csv and fleet_learning_experiment.csv.
"""

import argparse
import json
import random
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn"))
sys.path.insert(0, str(ROOT / "agents"))

warnings.filterwarnings("ignore", category=FutureWarning)

import torch  # noqa: E402

from disruptions import apply_incident, edge_at_fraction, edges_near  # noqa: E402
from feedback_loop import ClosedLoopExecutor, fleet_learning  # noqa: E402
from gnn_inference import GNNCostPredictor  # noqa: E402
from routing_solver import RoutingSolver, canonical_edges  # noqa: E402
from snapshot_disruptions import with_congestion_surge, with_rain  # noqa: E402

SNAPSHOT_DIR = ROOT / "dataset_kaggle" / "test"
OUT_DIR = ROOT / "agents" / "evaluation_results"

# The live app's solver-only presets.
PRIORITY_WEIGHTS = {
    "fastest": {"time": 0.7, "delay": 0.2, "carbon": 0.05, "ev": 0.05},
    "balanced": {"time": 0.25, "delay": 0.25, "carbon": 0.25, "ev": 0.25},
    "greenest": {"time": 0.1, "delay": 0.1, "carbon": 0.4, "ev": 0.4},
}
KINDS = ("none", "incident", "rain", "surge", "update")
CHANGE_AT = 0.30            # share of the promised time driven before the change
INCIDENT_AT = 0.65          # where on the route the incident is centred
SURGE_FROM = 0.45           # the route is surged from here to the destination
MIN_TRIP_S = 600.0          # shorter trips leave nothing to re-plan


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for kind, g in rows.groupby("kind", sort=False):
        switched = g[g["replan"]]
        out.append({
            "kind": kind,
            "trips": len(g),
            "road_ahead_slowed": round((g["slowdown_ahead"] > 1.01).mean(), 3),
            "mean_slowdown_ahead": round(g["slowdown_ahead"].mean(), 3),
            "replanned": round(g["replan"].mean(), 3),
            "late_15pct_without_replan": round((g["late_pct_without"] >= 0.15).mean(), 3),
            "late_15pct_with_loop": round((g["late_pct_with"] >= 0.15).mean(), 3),
            "mean_min_without_replan": round(g["time_without_s"].mean() / 60, 2),
            "mean_min_with_loop": round(g["time_with_s"].mean() / 60, 2),
            "mean_min_saved_all_trips": round(g["saved_time_s"].mean() / 60, 2),
            "mean_min_saved_when_replanned": round(switched["saved_time_s"].mean() / 60, 2) if len(switched) else 0.0,
            "mean_kg_saved_when_replanned": round(switched["saved_carbon_kg"].mean(), 3) if len(switched) else 0.0,
            "replans_that_ended_slower": round((switched["saved_time_s"] < 0).mean(), 3) if len(switched) else 0.0,
        })
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--date", default="2024-06-24")
    parser.add_argument("--hours", type=int, nargs="+", default=[9, 13, 21])
    parser.add_argument("--trips", type=int, default=120, help="trips in total, spread over the hours")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vehicles", type=int, default=5, help="vehicles per trip in the long-loop run")
    args = parser.parse_args()

    index = json.loads((SNAPSHOT_DIR / "_index.json").read_text())
    by_hour = {m["hour"]: name for name, m in index.items() if m["date"] == args.date}
    missing = [h for h in args.hours if h not in by_hour]
    if missing:
        raise SystemExit(f"No snapshot on {args.date} at hour(s) {missing}; available: {sorted(by_hour)}")

    def load(hour):
        return torch.load(SNAPSHOT_DIR / f"{by_hour[hour]}.pt", weights_only=False)

    predictor = GNNCostPredictor(checkpoint_dir=str(ROOT / "gnn_checkpoints_v3"))
    rng = random.Random(args.seed)
    per_hour = max(args.trips // len(args.hours), 1)
    rows, fleet_rows = [], []
    started = time.perf_counter()

    for hour in args.hours:
        snapshot = load(hour)
        costs = predictor.predict(snapshot)
        planner = RoutingSolver(costs)
        ce = canonical_edges(costs)
        edge_time = dict(zip(zip(ce["src"].tolist(), ce["dst"].tolist()), ce["travel_time_s"].tolist()))
        row_of = dict(zip(zip(ce["src"].tolist(), ce["dst"].tolist()), ce.index.tolist()))
        nodes = sorted(set(ce["src"]) | set(ce["dst"]))

        # Worlds every trip in this hour shares.
        shared = {"none": costs, "rain": predictor.predict(with_rain(snapshot))}
        later = [h for h in sorted(by_hour) if h > hour]
        if later:
            shared["update"] = predictor.predict(load(later[0]))
        shared_solvers = {k: RoutingSolver(w, reference_costs=costs) for k, w in shared.items()}

        made = 0
        while made < per_hour:
            source, target = rng.sample(nodes, 2)
            priority = list(PRIORITY_WEIGHTS)[made % len(PRIORITY_WEIGHTS)]
            weights = PRIORITY_WEIGHTS[priority]
            plan = planner.find_route(source, target, weights=weights)
            if not plan["success"] or plan["expected_travel_time_s"] < MIN_TRIP_S:
                continue
            path = plan["path"]
            trip = f"{hour:02d}h-{made:03d}"
            made += 1

            centre = edge_at_fraction(path, edge_time, INCIDENT_AT)
            incident = apply_incident(costs, edges_near(costs, centre, hops=1))
            first_surged = path.index(edge_at_fraction(path, edge_time, SURGE_FROM)[0])
            surge_rows = [row_of[e] for e in zip(path[first_surged:-1], path[first_surged + 1:])]
            worlds = {**shared, "incident": incident,
                      "surge": predictor.predict(with_congestion_surge(snapshot, surge_rows))}

            for kind in KINDS:
                if kind not in worlds:
                    continue   # no later snapshot that day
                report = ClosedLoopExecutor(
                    costs, worlds[kind], seed=f"{args.seed}|{trip}", use_llm=False,
                    planner=planner, updated=shared_solvers.get(kind),
                ).run(path, weights, priority=priority, disruption_after_fraction=CHANGE_AT, request_id=trip)
                decision, outcome = report["decision"], report["outcome"]
                without = report["counterfactual"] if decision["replan"] else outcome
                promised = report["promised"]["expected_time_s"]
                rows.append({
                    "trip": trip, "hour": hour, "priority": priority, "kind": kind,
                    "segments": len(path) - 1, "promised_s": promised,
                    "slowdown_ahead": report["slowdown_ahead"],
                    "replan": decision["replan"], "rule": decision["rule"],
                    "cost_gain_pct": decision.get("cost_gain_pct", 0.0),
                    "time_without_s": without["actual_time_s"], "time_with_s": outcome["actual_time_s"],
                    "carbon_without_kg": without["actual_carbon_kg"], "carbon_with_kg": outcome["actual_carbon_kg"],
                    "saved_time_s": report["saved"]["time_s"], "saved_carbon_kg": report["saved"]["carbon_kg"],
                    "late_pct_without": without["actual_time_s"] / promised - 1.0,
                    "late_pct_with": outcome["actual_time_s"] / promised - 1.0,
                })

            learned = fleet_learning(costs, incident, source, target, weights,
                                     n_vehicles=args.vehicles, seed=f"{args.seed}|{trip}")
            for t in learned["trips"]:
                fleet_rows.append({
                    "trip": trip, "hour": hour, "priority": priority, "vehicle": t["vehicle"],
                    "segments_corrected_before_planning": t["segments_corrected_before_planning"],
                    "actual_time_s": t["actual_time_s"], "no_feedback_time_s": t["no_feedback_time_s"],
                    "informed_time_s": t["informed_time_s"],
                    "same_as_no_feedback": t["same_as_no_feedback"], "same_as_informed": t["same_as_informed"],
                })
        print(f"  {hour:02d}:00  {made} trips done  ({time.perf_counter() - started:.0f}s)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "replan_experiment.csv", index=False)
    summary = summarise(results)
    summary.to_csv(OUT_DIR / "replan_summary.csv", index=False)
    fleet = pd.DataFrame(fleet_rows)
    fleet.to_csv(OUT_DIR / "fleet_learning_experiment.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(f"\nShort loop -- change reported to the planner at {CHANGE_AT:.0%} of the trip "
          f"({results['trip'].nunique()} trips, {args.date}, hours {args.hours})")
    print(summary.T.to_string(header=False))
    print("\nWhy the loop did or did not re-plan (trips per rule):")
    print(results.groupby(["kind", "rule"], sort=False).size().unstack(fill_value=0).to_string())

    print(f"\nLong loop -- incident never reported, {args.vehicles} vehicles in a row learn it from telemetry")
    per_vehicle = fleet.groupby("vehicle").agg(
        mean_min=("actual_time_s", lambda s: round(s.mean() / 60, 2)),
        no_feedback_min=("no_feedback_time_s", lambda s: round(s.mean() / 60, 2)),
        fully_informed_min=("informed_time_s", lambda s: round(s.mean() / 60, 2)),
        still_on_original_route=("same_as_no_feedback", lambda s: round(s.mean(), 3)),
        on_the_informed_route=("same_as_informed", lambda s: round(s.mean(), 3)),
    )
    print(per_vehicle.to_string())
    print(f"\nWrote {OUT_DIR / 'replan_experiment.csv'}, replan_summary.csv, fleet_learning_experiment.csv")


if __name__ == "__main__":
    main()
