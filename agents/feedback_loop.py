"""
feedback_loop.py

The loop from Execution back to routing (CLAUDE.md 11, Phase 4 of the
2026-10-11 plan). Until now the Execution agent drove a whole trip in
simulation and only then set replan_triggered -- a flag nothing acted on,
raised after the trip was already over. This module makes the loop real:

    plan -> drive -> conditions change part-way -> re-plan from where the
    vehicle is -> drive the rest -> telemetry -> corrected costs for the
    next plan

It is the Execution/Monitoring agent's job extended, not a seventh agent
(team decision 2026-10-07: the six agents in this repo only). The routing
is still RoutingSolver; the judgment call is still an LLM call in the same
"deterministic tool + LLM judgment" split as every other agent.

TWO LOOPS
---------
1. Short loop, within one trip -- ClosedLoopExecutor. The planner is told
   that conditions changed (a new snapshot, rain, an incident report). It
   compares finishing the planned route with the best route from the
   vehicle's current junction, both priced on the new conditions, and
   decides whether to switch.
2. Long loop, across trips -- FeedbackStore and fleet_learning(). Nobody
   tells the planner anything. Vehicles report how long each segment
   actually took; segments that keep taking longer than predicted get a
   correction factor, and the next vehicle is planned on corrected costs.

There is no live traffic in this project (CLAUDE.md 8.1). A "change" is a
second set of per-edge costs -- see agents/disruptions.py for the two kinds
(GNN-level and world-level) and what each one can honestly claim.

HOW THE RE-PLAN DECISION IS MADE (short loop)
---------------------------------------------
Deterministic rules first, in this order; the LLM is asked only when none
of them settles it:
 1. same_route  -- the best route from here IS the rest of the plan.
 2. ev_range    -- HARD rule. If finishing the plan is expected to need more
                   energy than the battery has left and the alternative
                   fits, switch. Never switch to a route that does not fit
                   when the plan does.
 3. dominated   -- never switch to a route that is expected to be both
                   slower and more polluting.
 4. small_gain  -- the alternative is under REPLAN_MIN_GAIN_PCT better on
                   the route's own weighted objectives: not worth turning a
                   vehicle around for.
 5. llm         -- weigh the time and carbon changes against the request's
                   priority. With the LLM off (or unavailable) the fallback
                   is "policy": switch, since rule 4 was passed.

WHY A LATE TRIP WITH NO NEW INFORMATION IS NOT RE-PLANNED
---------------------------------------------------------
If nothing about the roads has changed, the rest of a lowest-cost route is
still the lowest-cost route from any junction on it, so rule 1 fires. Time
already lost to ordinary delays cannot be bought back by re-routing; only
new information about the roads AHEAD can make a re-plan worth it. That is
why the old after-the-fact flag could never have led anywhere.

PAIRED COMPARISON
-----------------
Both futures (finish the plan / switch) are simulated, not just the chosen
one, with the same random numbers on every segment they share (each
segment's noise is seeded by the run seed and the segment itself). The
difference between the two outcomes is then the effect of the decision, not
of luck. Noise model: execution_agent.simulate_edge.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gnn"))

from execution_agent import EV_REFERENCE_KWH, TIME_JITTER_PCT, simulate_edge
from llm_client import align_ids, chat_json
from routing_solver import DELAY_EXTRA_TIME_RANGE, DELAY_SLOWDOWN, RoutingSolver, canonical_edges  # noqa: E402
from schemas import Vehicle

# Below this improvement on the route's own weighted cost, a switch is not
# worth making (and the LLM is not asked).
REPLAN_MIN_GAIN_PCT = 0.05

REPLAN_SYSTEM_PROMPT = """You are the Execution/Monitoring agent for a fleet of delivery vehicles.
A vehicle is part-way through a delivery and road conditions ahead have
changed. You are given two options from where the vehicle is now, both
already priced on the NEW conditions:
  continue_on_planned_route -- finish the route it was given
  switch_to_new_route       -- the best route from its current position
All times are in MINUTES and include typical delays. "switching_changes" is
the new route minus the planned one, so a NEGATIVE minutes value means the
switch saves time and a NEGATIVE co2_kg value means it saves carbon.

Decide whether to switch. Weigh the request's stated priority: a "fastest"
request should take a real time saving even at some carbon cost; a
"greenest" request should not accept noticeably more carbon unless the time
saving is large; "balanced" sits between. A switch that is worse on the
thing the request cares about most is not worth making.

Respond with ONLY this JSON object:
    {"decisions": {"<request_id>": {"replan": true, "reasoning": "one sentence"}}}
where "<request_id>" is replaced by the exact request_id string from the input.
"""


def _edge_rows(costs_df: pd.DataFrame) -> dict:
    """{(src, dst): row}, with the same choice among parallel roads as the solver."""
    return {(row.src, row.dst): row for row in canonical_edges(costs_df).itertuples(index=False)}


def _expected_time_s(row) -> float:
    return row.travel_time_s * (1.0 + DELAY_SLOWDOWN * row.delay_probability)


def drive(path: list, world: dict, believed: dict, seed, phase: str = "") -> dict:
    """
    Simulates driving `path` in `world` ({(src, dst): row} of what the roads
    are really like). `believed` is the same for what the planner thought
    they were like; each telemetry record carries that belief's expected
    time next to the actual one.

    Each segment draws its noise from its own generator, seeded by `seed`
    and the segment, so two paths simulated with the same seed get identical
    luck on every segment they share.
    """
    time_s = carbon_kg = ev_pct = 0.0
    delayed_edges = 0
    telemetry = []
    for u, v in zip(path[:-1], path[1:]):
        edge_time, edge_carbon, edge_ev, delayed = simulate_edge(world[(u, v)], random.Random(f"{seed}|{u}|{v}"))
        time_s += edge_time
        carbon_kg += edge_carbon
        ev_pct += edge_ev
        delayed_edges += delayed
        telemetry.append({
            "src": u, "dst": v, "phase": phase,
            "expected_time_s": _expected_time_s(believed[(u, v)]),
            "actual_time_s": edge_time,
            "delayed": bool(delayed),
        })
    return {"time_s": time_s, "carbon_kg": carbon_kg, "ev_energy_pct": ev_pct,
            "num_delayed_edges": delayed_edges, "telemetry": telemetry}


class ClosedLoopExecutor:
    """
    One trip, one change of conditions part-way, one re-plan decision.

    costs_before: the per-edge costs the route was planned on.
    costs_after:  the costs once conditions have changed.
    informed:     True  -- the planner receives costs_after when the change
                           happens and may re-plan (short loop).
                  False -- nobody tells the planner; the vehicle finishes its
                           plan in the changed world and only its telemetry
                           shows that something was wrong (long loop).
    planner, updated: optional ready-made solvers for costs_before and for
                  costs_after (the latter built with reference_costs=
                  costs_before), to save rebuilding them when many trips
                  share the same two worlds.
    """

    def __init__(self, costs_before: pd.DataFrame, costs_after: pd.DataFrame, model: str = None,
                 seed=0, use_llm: bool = True, informed: bool = True,
                 planner: RoutingSolver = None, updated: RoutingSolver = None):
        self.world_before = _edge_rows(costs_before)
        self.world_after = _edge_rows(costs_after)
        self.planner = planner or RoutingSolver(costs_before)
        # The new costs priced in the original plan's units -- see
        # RoutingSolver's reference_costs.
        self.updated = updated or RoutingSolver(
            costs_after, reference_costs=costs_before, algorithm=self.planner.algorithm)
        self.model = model
        self.seed = seed
        self.use_llm = use_llm
        self.informed = informed

    def _option(self, path: list, weights: dict) -> dict:
        totals = self.updated.path_totals(path)
        return {
            "path": path,
            "num_edges": totals["num_edges"],
            "expected_time_s": totals["expected_travel_time_s"],
            "expected_carbon_kg": totals["expected_carbon_kg"],
            "expected_ev_energy_pct": totals["expected_ev_energy_pct"],
            "planner_cost": self.updated.route_cost(path, weights),
        }

    def run(self, path: list, weights: dict, priority: str = "balanced", vehicle: Vehicle = None,
            disruption_after_fraction: float = 0.4, request_id: str = "trip",
            event: str = "Road conditions ahead have changed.") -> dict:
        """
        path, weights: the planned route and the weights it was planned with.
        disruption_after_fraction: when conditions change, as a share of the
        trip's promised (expected) time. The vehicle finishes the segment it
        is on, so the re-plan happens at a junction.
        """
        promised = self.planner.path_totals(path)
        promised_s = promised["expected_travel_time_s"]

        # 1. Drive the plan, in the world it was made for, until the change.
        edges_total = len(path) - 1
        done, elapsed_s = 0, 0.0
        before = {"time_s": 0.0, "carbon_kg": 0.0, "ev_energy_pct": 0.0, "num_delayed_edges": 0, "telemetry": []}
        while done < edges_total and elapsed_s < disruption_after_fraction * promised_s:
            step = drive(path[done:done + 2], self.world_before, self.world_before, self.seed, phase="before the change")
            for key in ("time_s", "carbon_kg", "ev_energy_pct", "num_delayed_edges"):
                before[key] += step[key]
            before["telemetry"] += step["telemetry"]
            elapsed_s = before["time_s"]
            done += 1

        report = {
            "request_id": request_id, "priority": priority, "event": event, "informed": self.informed,
            "weights": weights,
            "promised": {"expected_time_s": promised_s, "expected_carbon_kg": promised["expected_carbon_kg"],
                         "expected_ev_energy_pct": promised["expected_ev_energy_pct"]},
            "position": {"node": path[done], "edges_done": done, "edges_total": edges_total,
                         "elapsed_s": elapsed_s, "carbon_kg": before["carbon_kg"],
                         "ev_energy_pct": before["ev_energy_pct"]},
        }
        if done == edges_total:
            # Arrived before anything changed.
            return {**report, "disruption_reached": False, "final_path": path,
                    "decision": {"replan": False, "rule": "arrived",
                                 "reasoning": "The vehicle arrived before conditions changed."},
                    "outcome": self._outcome(before, None, vehicle), "counterfactual": None,
                    "saved": {"time_s": 0.0, "carbon_kg": 0.0, "ev_energy_pct": 0.0},
                    "telemetry": before["telemetry"]}

        # 2. The two options from here, both priced on the new conditions.
        current, target = path[done], path[-1]
        remaining = path[done:]
        stay = self._option(remaining, weights)
        found = self.updated.find_route(current, target, weights=weights)
        alternative = self._option(found["path"], weights) if found["success"] else None
        if alternative and alternative["path"] == remaining:
            alternative = None

        believed_rest_s = self.planner.path_totals(remaining)["expected_travel_time_s"]
        battery_left_pct = None
        if vehicle and vehicle.is_ev and vehicle.battery_capacity_kwh is not None:
            battery_left_pct = vehicle.battery_capacity_kwh / EV_REFERENCE_KWH * 100.0 - before["ev_energy_pct"]

        # 3. Decide.
        decision = self._decide(request_id, priority, event, elapsed_s, promised_s, stay, alternative, battery_left_pct)

        # 4. Drive both futures in the changed world, same luck on shared roads.
        believed = self.world_after if self.informed else self.world_before
        after_change = "after the change"
        stay_sim = drive(remaining, self.world_after, believed, self.seed, phase=after_change)
        alt_sim = (drive(alternative["path"], self.world_after, believed, self.seed, phase=after_change + ", new route")
                   if alternative else None)
        stay_outcome = self._outcome(before, stay_sim, vehicle)
        alt_outcome = self._outcome(before, alt_sim, vehicle) if alt_sim else None

        replanned = decision["replan"]
        chosen_sim, outcome, counterfactual = (
            (alt_sim, alt_outcome, stay_outcome) if replanned else (stay_sim, stay_outcome, alt_outcome))
        saved = {"time_s": 0.0, "carbon_kg": 0.0, "ev_energy_pct": 0.0}
        if replanned:
            saved = {
                "time_s": stay_outcome["actual_time_s"] - alt_outcome["actual_time_s"],
                "carbon_kg": stay_outcome["actual_carbon_kg"] - alt_outcome["actual_carbon_kg"],
                "ev_energy_pct": stay_outcome["actual_ev_energy_pct"] - alt_outcome["actual_ev_energy_pct"],
            }

        return {
            **report,
            "disruption_reached": True,
            # How much slower the rest of the plan is now expected to be: 1.0
            # means the change does not touch the road ahead.
            "slowdown_ahead": stay["expected_time_s"] / believed_rest_s if believed_rest_s > 0 else 1.0,
            "projected_deviation_pct": (elapsed_s + stay["expected_time_s"] - promised_s) / promised_s if promised_s > 0 else 0.0,
            "battery_left_pct": battery_left_pct,
            "stay": stay,
            "alternative": alternative,
            "decision": decision,
            "final_path": path[:done] + (alternative["path"] if replanned else remaining),
            "outcome": outcome,
            "counterfactual": counterfactual,
            "saved": saved,
            "telemetry": before["telemetry"] + chosen_sim["telemetry"],
        }

    @staticmethod
    def _outcome(before: dict, after: dict | None, vehicle: Vehicle | None) -> dict:
        after = after or {"time_s": 0.0, "carbon_kg": 0.0, "ev_energy_pct": 0.0, "num_delayed_edges": 0}
        ev_pct = before["ev_energy_pct"] + after["ev_energy_pct"]
        ev_violation = bool(
            vehicle and vehicle.is_ev and vehicle.battery_capacity_kwh is not None
            and ev_pct / 100.0 * EV_REFERENCE_KWH > vehicle.battery_capacity_kwh)
        return {
            "actual_time_s": before["time_s"] + after["time_s"],
            "actual_carbon_kg": before["carbon_kg"] + after["carbon_kg"],
            "actual_ev_energy_pct": ev_pct,
            "num_delayed_edges": before["num_delayed_edges"] + after["num_delayed_edges"],
            "ev_violation": ev_violation,
        }

    def _decide(self, request_id, priority, event, elapsed_s, promised_s, stay, alternative, battery_left_pct) -> dict:
        if not self.informed:
            return {"replan": False, "rule": "not_informed",
                    "reasoning": "The planner was not told that conditions changed, so the vehicle kept to its plan."}
        if alternative is None:
            return {"replan": False, "rule": "same_route",
                    "reasoning": "The best route from here is the one already planned."}

        time_delta_s = alternative["expected_time_s"] - stay["expected_time_s"]
        carbon_delta_kg = alternative["expected_carbon_kg"] - stay["expected_carbon_kg"]
        gain_pct = 1.0 - alternative["planner_cost"] / stay["planner_cost"] if stay["planner_cost"] > 0 else 0.0
        numbers = {"time_delta_s": time_delta_s, "carbon_delta_kg": carbon_delta_kg, "cost_gain_pct": gain_pct}

        if battery_left_pct is not None:
            stay_fits = stay["expected_ev_energy_pct"] <= battery_left_pct
            alt_fits = alternative["expected_ev_energy_pct"] <= battery_left_pct
            if not stay_fits and alt_fits:
                return {"replan": True, "rule": "ev_range", **numbers,
                        "reasoning": "Finishing the planned route is now expected to need more energy than the "
                                     "battery has left; the new route fits -- must re-plan."}
            if not stay_fits and not alt_fits:
                less = alternative["expected_ev_energy_pct"] < stay["expected_ev_energy_pct"]
                return {"replan": less, "rule": "ev_range", **numbers,
                        "reasoning": "Neither route is expected to fit the battery that is left; "
                                     + ("switched to the one that needs less energy."
                                        if less else "the planned one needs less energy, so kept it.")}
            if not alt_fits:
                return {"replan": False, "rule": "ev_range", **numbers,
                        "reasoning": "The new route would not fit the battery that is left; the planned one does."}

        if time_delta_s >= 0 and carbon_delta_kg >= 0:
            return {"replan": False, "rule": "dominated", **numbers,
                    "reasoning": "The new route is expected to be no faster and no cleaner than the planned one."}
        if gain_pct < REPLAN_MIN_GAIN_PCT:
            return {"replan": False, "rule": "small_gain", **numbers,
                    "reasoning": f"The new route is only {gain_pct:.1%} better on this delivery's objectives "
                                 f"({time_delta_s / 60:+.1f} min, {carbon_delta_kg:+.2f} kg CO2) -- under the "
                                 f"{REPLAN_MIN_GAIN_PCT:.0%} needed to justify switching."}

        policy = {"replan": True, "rule": "policy", **numbers,
                  "reasoning": f"The new route is {gain_pct:.0%} better on this delivery's objectives "
                               f"({time_delta_s / 60:+.1f} min, {carbon_delta_kg:+.2f} kg CO2)."}
        if not self.use_llm:
            return policy

        payload = {"trip": {
            "request_id": request_id,
            "priority": priority,
            "what_changed": event,
            "minutes_driven_so_far": round(elapsed_s / 60, 1),
            "promised_total_minutes": round(promised_s / 60, 1),
            "continue_on_planned_route": {
                "expected_total_minutes": round((elapsed_s + stay["expected_time_s"]) / 60, 1),
                "co2_kg_from_here": round(stay["expected_carbon_kg"], 3),
            },
            "switch_to_new_route": {
                "expected_total_minutes": round((elapsed_s + alternative["expected_time_s"]) / 60, 1),
                "co2_kg_from_here": round(alternative["expected_carbon_kg"], 3),
            },
            "switching_changes": {"minutes": round(time_delta_s / 60, 1), "co2_kg": round(carbon_delta_kg, 3)},
        }}
        messages = [
            {"role": "system", "content": REPLAN_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ]
        try:
            reply = chat_json(messages, model=self.model)
        except Exception as e:  # rate limit, bad reply, no key: a vehicle on the road still needs an answer
            return {**policy, "rule": "policy_fallback",
                    "reasoning": f"LLM unavailable ({type(e).__name__}); fell back to the fixed rule. " + policy["reasoning"]}
        answer = align_ids(reply.get("decisions", {}), [request_id]).get(request_id)
        if not isinstance(answer, dict) or "replan" not in answer:
            return {**policy, "rule": "policy_fallback",
                    "reasoning": "No usable LLM reply; fell back to the fixed rule. " + policy["reasoning"]}
        return {"replan": bool(answer["replan"]), "rule": "llm", **numbers,
                "reasoning": answer.get("reasoning") or "(the LLM gave no reason)"}


# ══════════════════════════════════════════════════════════════════════════
# LONG LOOP -- learning from telemetry
# ══════════════════════════════════════════════════════════════════════════

class FeedbackStore:
    """
    Per-segment correction factors learned from what vehicles experienced.

    For each segment it keeps the ratios actual time / expected time that
    vehicles reported. The factor is a shrunk mean,
        (prior_weight * 1.0 + sum of ratios) / (prior_weight + reports),
    so one bad run moves it only part of the way and it settles as reports
    accumulate. A factor within `deadband` of 1.0 is treated as 1.0, so that
    ordinary bad luck does not start steering routes. The default deadband
    is the largest factor ONE report can produce on a healthy road: the
    noise model's worst case is +10% jitter and a delay at its 2.0x maximum,
    a ratio of 2.2, which gives (2 + 2.2) / 3 = 1.4 -- so 0.4. A 4x incident
    reports a ratio of at least 2.6 and clears it on the first report.

    "Expected" must be the GNN's own expectation for the segment, not an
    already-corrected one -- the factor is applied to the GNN's costs.

    Known gap: a correction never expires. Once a segment is flagged,
    vehicles are routed around it, so no new reports arrive and it stays
    flagged after the road has cleared. Add an expiry time or an occasional
    probe before using this outside a simulation.
    """

    def __init__(self, prior_weight: float = 2.0, deadband: float = None):
        self.prior_weight = prior_weight
        if deadband is None:
            worst_healthy_ratio = (1.0 + TIME_JITTER_PCT) * DELAY_EXTRA_TIME_RANGE[1]
            deadband = (prior_weight + worst_healthy_ratio) / (prior_weight + 1.0) - 1.0
        self.deadband = deadband
        self.ratio_sum = defaultdict(float)
        self.reports = defaultdict(int)

    def record(self, telemetry: list) -> None:
        for t in telemetry:
            if t["expected_time_s"] > 0:
                edge = (t["src"], t["dst"])
                self.ratio_sum[edge] += t["actual_time_s"] / t["expected_time_s"]
                self.reports[edge] += 1

    def factor(self, edge: tuple) -> float:
        n = self.reports.get(edge, 0)
        if not n:
            return 1.0
        f = (self.prior_weight + self.ratio_sum[edge]) / (self.prior_weight + n)
        return f if abs(f - 1.0) > self.deadband else 1.0

    def corrections(self) -> dict:
        """{edge: factor} for every segment whose factor is outside the deadband."""
        return {edge: f for edge in self.reports if (f := self.factor(edge)) != 1.0}

    def corrected(self, costs_df: pd.DataFrame) -> pd.DataFrame:
        """
        costs_df with each corrected segment's time, carbon and EV energy
        multiplied by its factor (carbon and energy follow time, as in the
        Execution noise model). Delay probability is left alone.
        """
        corrections = self.corrections()
        if not corrections:
            return costs_df
        out = costs_df.copy()
        factors = pd.Series(list(zip(out["src"].tolist(), out["dst"].tolist())), index=out.index).map(corrections)
        factors = factors.fillna(1.0).to_numpy()
        for column in ("travel_time_s", "carbon_kg", "ev_energy_pct"):
            # Through float64: the GNN's frame is float32, and pandas refuses
            # to write the product back into a float32 column.
            out[column] = out[column].to_numpy(dtype=float) * factors
        return out

    def save(self, path) -> None:
        data = {"prior_weight": self.prior_weight, "deadband": self.deadband,
                "edges": [[u, v, self.ratio_sum[(u, v)], n] for (u, v), n in self.reports.items()]}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "FeedbackStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls(data["prior_weight"], data["deadband"])
        for u, v, ratio_sum, n in data["edges"]:
            store.ratio_sum[(u, v)] = ratio_sum
            store.reports[(u, v)] = n
        return store


def fleet_learning(costs_believed: pd.DataFrame, costs_true: pd.DataFrame, source, target, weights: dict,
                   n_vehicles: int = 6, seed=0, store: FeedbackStore = None) -> dict:
    """
    The long loop on one origin-destination pair. `n_vehicles` make the same
    trip one after another. The planner only ever has costs_believed (the
    GNN's predictions); the roads really behave like costs_true. Each
    vehicle is planned on the believed costs corrected by what earlier
    vehicles reported, drives in the true world, and reports its telemetry.

    Each trip is also simulated, with the same luck, on the route planned
    with no feedback and on the route a fully informed planner would take,
    so the three are comparable trip by trip.
    """
    store = store or FeedbackStore()
    believed = _edge_rows(costs_believed)
    world = _edge_rows(costs_true)
    informed_solver = RoutingSolver(costs_true, reference_costs=costs_believed)
    no_feedback_path = RoutingSolver(costs_believed).find_route(source, target, weights=weights)["path"]
    informed_path = informed_solver.find_route(source, target, weights=weights)["path"]

    trips = []
    for k in range(n_vehicles):
        corrections = store.corrections()
        planner = RoutingSolver(store.corrected(costs_believed), reference_costs=costs_believed)
        plan = planner.find_route(source, target, weights=weights)
        trip_seed = f"{seed}|vehicle{k}"
        sim = drive(plan["path"], world, believed, trip_seed)
        trips.append({
            "vehicle": k + 1,
            "path": plan["path"],
            "segments_corrected_before_planning": len(corrections),
            "planned_expected_time_s": plan["expected_travel_time_s"],
            # The same route's expected time on the roads as they really are.
            "true_expected_time_s": informed_solver.path_totals(plan["path"])["expected_travel_time_s"],
            "actual_time_s": sim["time_s"],
            "actual_carbon_kg": sim["carbon_kg"],
            "no_feedback_time_s": drive(no_feedback_path, world, believed, trip_seed)["time_s"],
            "informed_time_s": drive(informed_path, world, believed, trip_seed)["time_s"],
            "same_as_no_feedback": plan["path"] == no_feedback_path,
            "same_as_informed": plan["path"] == informed_path,
        })
        store.record(sim["telemetry"])

    return {"trips": trips, "store": store, "no_feedback_path": no_feedback_path, "informed_path": informed_path,
            "informed_expected_time_s": informed_solver.path_totals(informed_path)["expected_travel_time_s"]}


# ══════════════════════════════════════════════════════════════════════════
# SMOKE TEST -- runs against sample_costs/*.csv with an incident placed on
# the planned route. No torch, no GNN checkpoint and no LLM key needed.
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import glob

    from disruptions import apply_incident, edge_at_fraction, edges_near

    sys.stdout.reconfigure(encoding="utf-8")

    sample_costs_dir = Path(__file__).resolve().parent.parent / "sample_costs"
    csv_files = sorted(glob.glob(str(sample_costs_dir / "*.csv")))
    if not csv_files:
        raise SystemExit(f"No sample cost CSVs found in {sample_costs_dir}")
    costs = pd.read_csv(csv_files[0])
    print(f"Loaded {csv_files[0]} ({len(costs)} edges)")

    weights = {"time": 0.7, "delay": 0.2, "carbon": 0.05, "ev": 0.05}
    planner = RoutingSolver(costs)
    nodes = pd.unique(costs[["src", "dst"]].values.ravel())
    plan = planner.find_route(int(nodes[100]), int(nodes[6000]), weights=weights)
    path = plan["path"]
    print(f"Planned: {plan['num_edges']} segments, expected {plan['expected_travel_time_s'] / 60:.1f} min")

    edge_time = {(r.src, r.dst): r.travel_time_s for r in canonical_edges(costs).itertuples(index=False)}
    centre = edge_at_fraction(path, edge_time, 0.7)
    blocked = edges_near(costs, centre, hops=1)
    after = apply_incident(costs, blocked)
    print(f"Incident: {len(blocked)} segments around {centre} slowed 4x, 40% of the way into the trip")

    for informed in (True, False):
        rep = ClosedLoopExecutor(costs, after, seed=0, use_llm=False, informed=informed, planner=planner).run(
            path, weights, priority="fastest")
        d, o, c = rep["decision"], rep["outcome"], rep["counterfactual"]
        print(f"\n{'Reported to the planner' if informed else 'NOT reported'}: "
              f"{'RE-PLAN' if d['replan'] else 'continue'} [{d['rule']}] -- {d['reasoning']}")
        print(f"  arrived after {o['actual_time_s'] / 60:.1f} min"
              + (f" (other option: {c['actual_time_s'] / 60:.1f} min)" if c else "")
              + f", promised {rep['promised']['expected_time_s'] / 60:.1f} min")

    print("\nLong loop: the incident is never reported; vehicles learn it from telemetry")
    learned = fleet_learning(costs, after, path[0], path[-1], weights, n_vehicles=6, seed=0)
    for t in learned["trips"]:
        print(f"  vehicle {t['vehicle']}: {t['actual_time_s'] / 60:5.1f} min "
              f"(no feedback {t['no_feedback_time_s'] / 60:5.1f}, fully informed {t['informed_time_s'] / 60:5.1f}) "
              f"corrected segments known: {t['segments_corrected_before_planning']}"
              + ("  <- same route as a fully informed planner" if t["same_as_informed"] else ""))
