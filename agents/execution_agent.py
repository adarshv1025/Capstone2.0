"""
execution_agent.py

Phase 6: Execution/Monitoring Agent (CLAUDE.md roster #5). Simulates
actually driving the finalized plan against the GNN's per-edge predictions,
with injected noise standing in for real-world deviation from what the GNN
predicted, then flags routes whose simulated outcome diverges enough from
plan to warrant a re-plan.

Grounding: costs_df (the GNN's own per-edge predictions) is the only
"truth" available locally -- there's no live telemetry to simulate against,
so noise is injected on top of the GNN's own numbers to stand in for
real-world variance. See NOISE MODEL below for the exact numbers and why.

Split, same pattern as the rest of this layer (CLAUDE.md 3.2):
 - Deterministic: the simulation itself, and one HARD rule -- an EV whose
   simulated actual consumption exceeds its battery is always a re-plan
   trigger, no judgment call about it (a stranded EV is a hard failure, not
   a trade-off, matching the pattern already used for the Carbon
   Optimizer's carbon_delta_kg >= 0 hard rule in Phase 5).
 - LLM judgment: for routes with a noticeable but non-catastrophic time
   deviation, decide (given the request's stated priority) whether it's
   bad enough to actually interrupt and re-plan, or just normal variance
   worth logging and continuing.

Actually triggering a new routing pass in response to replan_triggered is
out of scope here -- this agent's job (per the roster) is to simulate and
flag, not to re-plan. A later phase or the eval harness consumes the flag.

NOISE MODEL (per edge, applied on top of costs_df's predicted values)
-----------------------------------------------------------------------
1. Baseline prediction error: actual_time = predicted_time * (1 + U(-10%, +10%)).
   The GNN's travel_time R^2 is 0.9489 (CLAUDE.md 2.4) -- not perfect --
   so a +-10% jitter is a reasonable stand-in for normal residual error.
2. Delay events: drawn as a Bernoulli trial using the GNN's OWN predicted
   delay_probability for that edge (principled: the simulation's "did a
   delay actually happen" uses the model's own uncertainty estimate rather
   than an arbitrary constant). If delayed, that edge's time is further
   multiplied by U(1.3, 2.0) -- a delay adds 30-100% more time to just that
   segment, not the whole route.
3. Carbon and EV energy scale with the realized time ratio for that edge
   (actual/predicted travel time) rather than getting independent noise --
   physically, more time on a segment (idling, congestion) means
   proportionally more emissions/energy, so tying them to the same jitter
   is more defensible than inventing a third independent noise parameter.
"""

import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gnn"))

from llm_client import chat_json
from schemas import RouteResult, Vehicle

TIME_JITTER_PCT = 0.10               # +-10% baseline prediction noise per edge
DELAY_EXTRA_TIME_RANGE = (1.3, 2.0)  # extra multiplier applied when an edge's delay actually fires
TIME_DEVIATION_FLOOR_PCT = 0.15      # below this magnitude, treat as normal variance -- don't even ask the LLM
EV_REFERENCE_KWH = 50.0              # ev_energy_pct is % of this reference battery, per CLAUDE.md 2.1

REFLECT_SYSTEM_PROMPT = """You are the Execution/Monitoring agent for a fleet of delivery vehicles.
You are given a list of routes that, once actually driven (simulated
against real-world variance), deviated from their planned travel time by a
noticeable amount. time_deviation_pct is (actual - planned) / planned, so a
POSITIVE value means the trip took LONGER than planned (worse); a NEGATIVE
value means it finished faster than planned (better).

For EACH route, decide whether this deviation is bad enough to interrupt
the vehicle and trigger a re-plan, or whether it's within normal variance
to just log and continue. Weigh the request's stated priority -- a
"fastest" request should have a much lower tolerance for running late than
a "greenest" or "balanced" one. A negative time_deviation_pct (finished
early) should essentially never trigger a re-plan regardless of priority.

Respond with ONLY this JSON object:
    {"decisions": {"req-0": {"replan": true, "reasoning": "one sentence"}}}
"""


class ExecutionAgent:
    def __init__(self, costs_df: pd.DataFrame, model: str = None, seed: int = None):
        self.edge_costs = {(row.src, row.dst): row for row in costs_df.itertuples(index=False)}
        self.model = model
        # A seeded RNG makes simulate_route() reproducible for testing;
        # seed=None (default) gives a fresh draw each run, as a real
        # execution simulation should.
        self.rng = random.Random(seed)

    def simulate_route(self, result: RouteResult) -> dict:
        """
        Replays one route's edges against costs_df with the noise model
        described in the module docstring. Returns actual totals and how
        many edges actually hit a simulated delay.
        """
        actual_time = 0.0
        actual_carbon = 0.0
        actual_ev = 0.0
        num_delayed = 0

        edges = list(zip(result.path[:-1], result.path[1:]))
        for u, v in edges:
            row = self.edge_costs.get((u, v))
            if row is None:
                continue  # shouldn't happen -- the route came from this same costs_df

            edge_time = row.travel_time_s * (1.0 + self.rng.uniform(-TIME_JITTER_PCT, TIME_JITTER_PCT))

            if self.rng.random() < row.delay_probability:
                edge_time *= self.rng.uniform(*DELAY_EXTRA_TIME_RANGE)
                num_delayed += 1

            time_ratio = edge_time / row.travel_time_s if row.travel_time_s > 0 else 1.0
            actual_time += edge_time
            actual_carbon += row.carbon_kg * time_ratio
            actual_ev += row.ev_energy_pct * time_ratio

        return {
            "actual_time_s": actual_time,
            "actual_carbon_kg": actual_carbon,
            "actual_ev_energy_pct": actual_ev,
            "num_delayed_edges": num_delayed,
            "num_edges": len(edges),
        }

    def _hard_ev_violation(self, sim: dict, vehicle: Vehicle | None) -> bool:
        if not vehicle or not vehicle.is_ev or vehicle.battery_capacity_kwh is None:
            return False
        actual_kwh = (sim["actual_ev_energy_pct"] / 100.0) * EV_REFERENCE_KWH
        return actual_kwh > vehicle.battery_capacity_kwh

    def run(self, results: list[RouteResult], vehicles_by_id: dict[str, Vehicle]) -> list[dict]:
        """
        Simulates every successful route once. Returns one report dict per
        successful route: the simulated outcome, time_deviation_pct, and
        the replan_triggered decision (+ reasoning).

        Batches the LLM reflection into ONE call across every route that
        needs one, not one call per route -- the same per-item-call scaling
        problem hit and fixed in the Coordinator (Phase 4) and Carbon
        Optimizer (Phase 5) would recur here otherwise.
        """
        reports = []
        needs_reflection = []

        for r in results:
            if not r.success:
                continue
            sim = self.simulate_route(r)
            time_deviation_pct = (
                (sim["actual_time_s"] - r.total_travel_time_s) / r.total_travel_time_s
                if r.total_travel_time_s > 0 else 0.0
            )
            hard_ev_violation = self._hard_ev_violation(sim, vehicles_by_id.get(r.vehicle_id))

            report = {
                "request_id": r.request_id,
                "vehicle_id": r.vehicle_id,
                "priority": r.priority,
                **sim,
                "planned_time_s": r.total_travel_time_s,
                "time_deviation_pct": time_deviation_pct,
                "hard_ev_violation": hard_ev_violation,
                "replan_triggered": hard_ev_violation,
                "reasoning": (
                    "Simulated EV energy use exceeds the vehicle's battery capacity -- must re-plan."
                    if hard_ev_violation else None
                ),
            }
            reports.append(report)

            if not hard_ev_violation and abs(time_deviation_pct) >= TIME_DEVIATION_FLOOR_PCT:
                needs_reflection.append(report)

        if needs_reflection:
            decisions = self._reflect(needs_reflection)
            for report in needs_reflection:
                decision = decisions.get(report["request_id"], {})
                report["replan_triggered"] = bool(decision.get("replan", False))
                report["reasoning"] = decision.get("reasoning") or "no reflection response -- defaulted to no re-plan"

        return reports

    def _reflect(self, reports: list[dict]) -> dict:
        payload = {
            "routes": [
                {
                    "request_id": r["request_id"],
                    "priority": r["priority"],
                    "planned_time_s": r["planned_time_s"],
                    "actual_time_s": r["actual_time_s"],
                    "time_deviation_pct": r["time_deviation_pct"],
                    "num_delayed_edges": r["num_delayed_edges"],
                    "num_edges": r["num_edges"],
                }
                for r in reports
            ]
        }
        messages = [
            {"role": "system", "content": REFLECT_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ]
        try:
            result = chat_json(messages, model=self.model)
        except ValueError:
            return {}
        return result.get("decisions", {})


# ══════════════════════════════════════════════════════════════════════════
# SMOKE TEST -- routes a small batch, then simulates execution. seed=0 makes
# the noise draws reproducible across runs. Requires a working LLM key in
# .env for the reflection step (routes with a small enough deviation skip it).
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import glob

    # LLM output can contain Unicode punctuation (e.g. non-breaking hyphens)
    # that Windows' default console codepage can't print -- force utf-8.
    sys.stdout.reconfigure(encoding="utf-8")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from route_agent import RouteAgent
    from schemas import DeliveryRequest

    sample_costs_dir = Path(__file__).resolve().parent.parent / "sample_costs"
    csv_files = sorted(glob.glob(str(sample_costs_dir / "*.csv")))
    if not csv_files:
        raise SystemExit(f"No sample cost CSVs found in {sample_costs_dir}")

    costs_df = pd.read_csv(csv_files[0])
    print(f"Loaded {csv_files[0]} ({len(costs_df)} edges)")

    nodes = pd.unique(costs_df[["src", "dst"]].values.ravel())
    pairs = [(0, 2000), (100, 6000), (200, 7000)]
    priorities = ["fastest", "balanced", "greenest"]

    requests = [
        DeliveryRequest(request_id=f"req-{i}", origin=int(nodes[a]), destination=int(nodes[b]), priority=p)
        for i, ((a, b), p) in enumerate(zip(pairs, priorities))
    ]
    vehicles_by_id = {f"veh-{i}": Vehicle(vehicle_id=f"veh-{i}") for i in range(len(requests))}
    for req, veh_id in zip(requests, vehicles_by_id):
        req.vehicle_id = veh_id

    route_agent = RouteAgent(costs_df)
    results = [route_agent.route(r) for r in requests]

    print("\nPlanned:")
    for r in results:
        print(f"  {r.request_id} priority={r.priority}  planned_time={r.total_travel_time_s:.1f}s")

    executor = ExecutionAgent(costs_df, seed=0)
    reports = executor.run(results, vehicles_by_id)

    print("\nSimulated execution:")
    for rep in reports:
        flag = "REPLAN" if rep["replan_triggered"] else "ok"
        print(
            f"  {rep['request_id']} [{flag}] planned={rep['planned_time_s']:.1f}s "
            f"actual={rep['actual_time_s']:.1f}s (dev={rep['time_deviation_pct']:+.1%}) "
            f"delayed_edges={rep['num_delayed_edges']}/{rep['num_edges']}"
        )
        if rep["reasoning"]:
            print(f"      reasoning: {rep['reasoning']}")
