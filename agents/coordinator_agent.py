"""
coordinator_agent.py

Phase 4: Coordinator/Negotiation Agent (CLAUDE.md roster #3) -- the
genuinely "decentralized" piece of the framework.

Conflict DETECTION is deterministic grounding (shared-edge overload counts,
EV battery-budget math -- pure facts, no judgment needed). Conflict
RESOLUTION is where the LLM's judgment comes in (CLAUDE.md 3.2 explicitly
names "negotiating trade-offs between competing vehicles" as an LLM task):
given a conflict, the LLM decides which requests give ground and how, and
the Coordinator deterministically re-solves those routes with
RoutingSolver. This loops until no conflicts remain or max_rounds is hit.

Two conflict types:
  - edge_overload: more than `edge_capacity` vehicles routed over the same
    directed edge. The LLM picks which request_ids give up the edge; the
    Coordinator penalizes that edge (RoutingSolver.find_route's
    penalized_edges) and re-solves those requests with their ORIGINAL
    weights_used -- the objective priority doesn't change, only the path
    does.
  - ev_range: an EV vehicle's route consumes more energy than its actual
    battery budget. `ev_energy_pct` is defined (CLAUDE.md 2.1) as % of a
    reference 50kWh battery, so it's rescaled against the vehicle's real
    battery_capacity_kwh to check feasibility. The LLM picks an adjusted
    {time, delay, carbon, ev} weights dict biased toward energy efficiency
    -- this IS a genuine trade-off judgment (how much time/carbon to give
    up for feasibility), unlike edge_overload's fixed mechanism.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gnn"))
from routing_solver import RoutingSolver  # noqa: E402

from llm_client import chat_json
from route_agent import build_route_result
from schemas import DeliveryRequest, RouteResult, Vehicle

NEGOTIATION_SYSTEM_PROMPT = """You are the Coordinator for a fleet of delivery vehicles sharing a road
network. Each round you are given ALL of the current conflicts of ONE type
at once, as a single JSON object, and must decide how to resolve all of
them together in one response (not one-by-one) -- a request that appears in
several conflicts should get one consistent decision, not contradictory ones.

Conflict type "edge_overload": the JSON's "conflict_groups" list has one
entry per distinct set of requests that are overloading road segments
together, with how many segments they share (shared_edge_count) and each
request's id, priority, and current time/carbon totals. Across ALL of
these groups, decide which request_ids should give up their current path
and be re-routed around the segments they're overloading -- a
fairness/negotiation call (e.g. weigh stated priority, how much extra cost
rerouting likely costs each one, avoid rerouting the same request for every
group it appears in if rerouting it once would clear several). Respond
with ONLY this JSON object:
    {"reroute_request_ids": ["req-2", "req-3"], "reasoning": "one sentence"}

Conflict type "ev_range": the JSON's "violations" list has one entry per
request whose route consumes more EV battery energy than its vehicle's
actual capacity allows (current {time, delay, carbon, ev} weights and how
much the route overshoots the battery budget by). For EACH request_id,
decide a new weights dict that shifts enough weight onto "ev" to fix
feasibility, trading off against the other three as little as necessary.
Respond with ONLY this JSON object:
    {"weight_adjustments": {"req-2": {"time": 0.1, "delay": 0.1, "carbon": 0.2, "ev": 0.6}}, "reasoning": "one sentence"}

Always respond with ONLY the single JSON object for the conflict's type, no
other text.
"""


class CoordinatorAgent:
    def __init__(
        self,
        costs_df: pd.DataFrame,
        vehicles: list[Vehicle],
        model: str = None,
        edge_capacity: int = 2,
        max_rounds: int = 3,
        ev_reference_kwh: float = 50.0,
        penalty_factor: float = 5.0,
    ):
        self.solver = RoutingSolver(costs_df)
        self.vehicles_by_id = {v.vehicle_id: v for v in vehicles}
        self.model = model
        self.edge_capacity = edge_capacity
        self.max_rounds = max_rounds
        self.ev_reference_kwh = ev_reference_kwh
        self.penalty_factor = penalty_factor
        # Set of (src, dst) edges penalized by the most recent resolve()
        # call -- exposed so a later stage (the Carbon Optimizer, Phase 5)
        # can avoid re-solving a route straight back onto an edge this
        # Coordinator already worked to route around.
        self.penalized_edges: set = set()

    def _detect_conflicts(self, results: list[RouteResult]) -> tuple[dict, list[str]]:
        edge_usage: dict[tuple, list[str]] = defaultdict(list)
        ev_violations: list[str] = []

        for r in results:
            if not r.success:
                continue
            for edge in zip(r.path[:-1], r.path[1:]):
                edge_usage[edge].append(r.request_id)

            vehicle = self.vehicles_by_id.get(r.vehicle_id)
            if vehicle and vehicle.is_ev and vehicle.battery_capacity_kwh is not None:
                consumed_kwh = (r.total_ev_energy_pct / 100.0) * self.ev_reference_kwh
                if consumed_kwh > vehicle.battery_capacity_kwh:
                    ev_violations.append(r.request_id)

        overloaded_edges = {edge: reqs for edge, reqs in edge_usage.items() if len(reqs) > self.edge_capacity}
        return overloaded_edges, ev_violations

    def _negotiate(self, conflict: dict) -> dict:
        messages = [
            {"role": "system", "content": NEGOTIATION_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(conflict)},
        ]
        try:
            return chat_json(messages, model=self.model)
        except ValueError:
            return {}

    def resolve(
        self, results: list[RouteResult], requests_by_id: dict[str, DeliveryRequest]
    ) -> tuple[list[RouteResult], int, bool]:
        """
        Iteratively re-routes conflicting requests until no conflicts
        remain or max_rounds is hit. Returns (final_results, rounds_used,
        fully_resolved). rounds_used feeds the Phase 8 evaluation harness's
        "negotiation rounds needed" metric.

        fully_resolved is False when max_rounds was exhausted with
        conflicts still outstanding -- this can be a genuinely infeasible
        request, not just a tuning failure. E.g. an EV violation's only
        lever is reweighting toward "ev", but testing found
        ev_energy_pct correlates strongly with route distance/time in this
        dataset (unlike delay_probability, which is segment-specific) --
        so for a battery too small for the trip's shortest possible path,
        no amount of reweighting finds a feasible route, and callers (the
        eval harness's EV feasibility rate metric in particular) need to
        see that honestly rather than have it silently swallowed.

        Each round issues at most 2 LLM calls total (one for ALL
        edge_overload conflicts combined, one for ALL ev_range conflicts
        combined) regardless of how many individual conflicts exist --
        batching them into one negotiation per type per round, rather than
        one call per conflict, keeps this from blowing through API rate
        limits on a busy batch and gives the LLM a coherent view instead of
        fragmented per-edge decisions.
        """
        results_by_id = {r.request_id: r for r in results}
        penalized_edges: set = set()
        rounds_used = 0

        for round_num in range(1, self.max_rounds + 1):
            overloaded_edges, ev_violations = self._detect_conflicts(list(results_by_id.values()))
            if not overloaded_edges and not ev_violations:
                self.penalized_edges = penalized_edges
                return list(results_by_id.values()), rounds_used, True
            rounds_used = round_num

            if overloaded_edges:
                # Group edges by their exact set of competing request_ids:
                # routes that overlap for a whole shared corridor otherwise
                # produce one near-duplicate entry per edge (a 21-edge
                # overlap between the same 3 requests is one decision, not
                # 21), which bloats the negotiation payload for no benefit
                # and can blow past a rate-limited provider's per-request
                # token cap on a busy batch.
                edge_groups: dict[frozenset, list[tuple]] = defaultdict(list)
                for edge, request_ids in overloaded_edges.items():
                    edge_groups[frozenset(request_ids)].append(edge)
                # Still cap group count, not just per-group edge count: a
                # batch with many genuinely distinct overlapping pairs (not
                # just one big shared corridor) can produce many groups.
                # Uncapped work (penalizing every edge, and the fallback
                # reroute decision below) still covers all of them
                # regardless of what fits in the prompt.
                top_groups = sorted(edge_groups.items(), key=lambda kv: -len(kv[1]))[:30]

                conflict = {
                    "type": "edge_overload",
                    "conflict_groups": [
                        {
                            "request_ids": list(request_ids),
                            "shared_edge_count": len(edges),
                            "requests": [
                                {
                                    "request_id": rid,
                                    "priority": results_by_id[rid].priority,
                                    "total_travel_time_s": results_by_id[rid].total_travel_time_s,
                                    "total_carbon_kg": results_by_id[rid].total_carbon_kg,
                                }
                                for rid in request_ids
                            ],
                        }
                        for request_ids, edges in top_groups
                    ],
                }
                decision = self._negotiate(conflict)
                # Fallback if the LLM call/parse failed: keep the first
                # `edge_capacity` requests on each edge, reroute the rest.
                fallback_reroute_ids = {
                    rid for request_ids in overloaded_edges.values() for rid in request_ids[self.edge_capacity:]
                }
                reroute_ids = decision.get("reroute_request_ids", list(fallback_reroute_ids))
                penalized_edges.update(overloaded_edges.keys())

                for rid in reroute_ids:
                    request = requests_by_id.get(rid)
                    if request is None or rid not in results_by_id:
                        continue
                    weights = results_by_id[rid].weights_used
                    solver_result = self.solver.find_route(
                        request.origin, request.destination,
                        weights=weights, penalized_edges=penalized_edges, penalty_factor=self.penalty_factor,
                    )
                    results_by_id[rid] = build_route_result(
                        request, solver_result, weights, decision.get("reasoning")
                    )

            if ev_violations:
                conflict = {
                    "type": "ev_range",
                    "violations": [
                        {
                            "request_id": rid,
                            "current_weights": results_by_id[rid].weights_used,
                            "route_ev_energy_pct": results_by_id[rid].total_ev_energy_pct,
                            "vehicle_battery_kwh": (
                                self.vehicles_by_id[results_by_id[rid].vehicle_id].battery_capacity_kwh
                                if results_by_id[rid].vehicle_id in self.vehicles_by_id else None
                            ),
                        }
                        for rid in ev_violations
                    ],
                }
                decision = self._negotiate(conflict)
                weight_adjustments = decision.get("weight_adjustments", {})

                for rid in ev_violations:
                    request = requests_by_id.get(rid)
                    if request is None:
                        continue
                    weights = weight_adjustments.get(rid, {"time": 0.1, "delay": 0.1, "carbon": 0.2, "ev": 0.6})
                    solver_result = self.solver.find_route(
                        request.origin, request.destination,
                        weights=weights, penalized_edges=penalized_edges, penalty_factor=self.penalty_factor,
                    )
                    results_by_id[rid] = build_route_result(
                        request, solver_result, weights, decision.get("reasoning")
                    )

        overloaded_edges, ev_violations = self._detect_conflicts(list(results_by_id.values()))
        fully_resolved = not overloaded_edges and not ev_violations
        self.penalized_edges = penalized_edges
        return list(results_by_id.values()), rounds_used, fully_resolved


# ══════════════════════════════════════════════════════════════════════════
# SMOKE TEST -- forces a real edge_overload conflict (same origin/dest for
# 3 requests, edge_capacity=2) and a real ev_range conflict (a tiny-battery
# EV). Requires a working LLM key in .env.
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import glob

    # LLM output can contain Unicode punctuation (e.g. non-breaking hyphens)
    # that Windows' default console codepage can't print -- force utf-8.
    sys.stdout.reconfigure(encoding="utf-8")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from route_agent import RouteAgent

    sample_costs_dir = Path(__file__).resolve().parent.parent / "sample_costs"
    csv_files = sorted(glob.glob(str(sample_costs_dir / "*.csv")))
    if not csv_files:
        raise SystemExit(f"No sample cost CSVs found in {sample_costs_dir}")

    costs_df = pd.read_csv(csv_files[0])
    print(f"Loaded {csv_files[0]} ({len(costs_df)} edges)")

    nodes = pd.unique(costs_df[["src", "dst"]].values.ravel())
    shared_origin, shared_destination = int(nodes[0]), int(nodes[-1])
    ev_origin, ev_destination = int(nodes[100]), int(nodes[6000])

    # req-0/1/2 share one origin/destination -- with edge_capacity=2 that's
    # a guaranteed edge_overload conflict. req-3 is a separate, unrelated
    # trip whose default route uses ~9.8% of a 50kWh battery (~4.9kWh); a
    # 4.8kWh vehicle can't quite make it on the default route but CAN on a
    # more energy-conscious one -- a genuinely resolvable ev_range conflict,
    # kept independent of the edge_overload trio so the two conflict types
    # don't compound (testing found EV energy correlates strongly with
    # route distance here, so forcing an edge-avoidance detour AND an
    # energy-efficient route on the same request can be jointly infeasible
    # -- see the fully_resolved note on CoordinatorAgent.resolve).
    requests = [
        DeliveryRequest(request_id=f"req-{i}", origin=shared_origin, destination=shared_destination, priority="balanced")
        for i in range(3)
    ] + [
        DeliveryRequest(request_id="req-3", origin=ev_origin, destination=ev_destination, priority="fastest")
    ]
    vehicles = [
        Vehicle(vehicle_id="veh-0", current_location=shared_origin),
        Vehicle(vehicle_id="veh-1", current_location=shared_origin),
        Vehicle(vehicle_id="veh-2", current_location=shared_origin),
        Vehicle(vehicle_id="veh-3", is_ev=True, battery_capacity_kwh=4.8, current_location=ev_origin),
    ]
    requests_by_id = {}
    for req, veh in zip(requests, vehicles):
        req.vehicle_id = veh.vehicle_id
        requests_by_id[req.request_id] = req

    route_agent = RouteAgent(costs_df)
    initial_results = [route_agent.route(r) for r in requests]

    print("\nBefore negotiation:")
    for r in initial_results:
        print(f"  {r.request_id} -> {r.vehicle_id}  edges={r.num_edges}  ev={r.total_ev_energy_pct:.2f}%")

    coordinator = CoordinatorAgent(costs_df, vehicles, edge_capacity=2)
    final_results, rounds_used, fully_resolved = coordinator.resolve(initial_results, requests_by_id)

    print(f"\nAfter negotiation ({rounds_used} round(s) used, fully_resolved={fully_resolved}):")
    for r in final_results:
        status = "OK" if r.success else f"FAILED ({r.reason})"
        print(f"  {r.request_id} -> {r.vehicle_id}  {status}  edges={r.num_edges}  ev={r.total_ev_energy_pct:.2f}%")
        print(f"      reasoning: {r.reasoning}")
