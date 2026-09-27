"""
carbon_optimizer_agent.py

Phase 5: Carbon-Aware Optimizer (CLAUDE.md roster #4) -- a critic with a
self-reflection loop that reviews the Coordinator's finalized plan and
pushes greener swaps where the time cost is acceptable.

Three steps, matching the "deterministic tools ground, LLM judges" pattern
(CLAUDE.md 3.2) used by every other agent in this layer:
  1. PROPOSE (LLM): for each successful route, propose a candidate
     {time, delay, carbon, ev} weights dict shifted toward lower carbon.
  2. GROUND (deterministic): re-solve each candidate with RoutingSolver and
     compute the real time/carbon deltas vs. the original route.
  3. REFLECT (LLM): given those deltas and the request's stated priority,
     decide whether the trade is actually worth it. This is the
     self-reflection step -- the agent looks at the real outcome of its own
     proposal, not just the proposal itself, before committing to it.

There's no hardcoded acceptance formula (e.g. "reject if time increases
>10%") -- that would just be moving the judgment call into code instead of
leaving it as genuine LLM judgment, contrary to 3.2. The reflect prompt
gives rough guidance for what "usually" counts as worth it, but the LLM
decides per-request based on priority, not a fixed threshold.

Like the Coordinator (Phase 4), both LLM steps are batched into ONE call
across the whole set of routes rather than one call per route -- the same
per-item-call scaling problem that blew through Groq's rate limit there
would recur here with 2 calls x N routes otherwise.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gnn"))
from routing_solver import RoutingSolver  # noqa: E402

from llm_client import chat_json
from route_agent import build_route_result
from schemas import DeliveryRequest, RouteResult

PROPOSE_SYSTEM_PROMPT = """You are the Carbon-Aware Optimizer for a fleet of delivery vehicles. You
review a finalized routing plan and look for opportunities to reduce carbon
emissions.

You are given a list of routes, each with its request_id, stated priority,
and current {time, delay, carbon, ev} weights. For EACH one, propose a
candidate weights dict shifted toward "carbon" -- but keep it sensible for
the stated priority (e.g. don't propose 100% carbon for a "fastest"
request; a modest shift is more realistic there than for a "greenest" one).
If the current weights already heavily favor carbon, you may propose the
same weights unchanged -- no forced change.

Respond with ONLY this JSON object:
    {"candidates": {"req-0": {"time": 0.2, "delay": 0.1, "carbon": 0.6, "ev": 0.1}}, "reasoning": "one sentence"}
"""

REFLECT_SYSTEM_PROMPT = """You are the Carbon-Aware Optimizer performing self-reflection on candidate
greener routes you proposed a moment ago. For EACH request below you're
given its stated priority and the REAL outcome of solving your candidate,
as time_delta_s and carbon_delta_kg -- both computed as (candidate minus
original), NOT as a percent-saved or already-signed-for-improvement value.
That means: a POSITIVE time_delta_s means the candidate is SLOWER (worse);
a POSITIVE carbon_delta_kg means the candidate emits MORE carbon (worse); a
NEGATIVE carbon_delta_kg means it emits LESS carbon (better -- this is the
only outcome that can justify accepting a slower candidate). Read the sign
carefully before deciding -- do not assume the candidate improved on carbon
just because it was proposed as a "greener" option; check the actual delta.

Decide, per request, whether the trade is actually worth taking. As rough
guidance: a small time increase (roughly under 10%) is usually worth almost
any real (negative carbon_delta_kg) carbon reduction; a much larger time
increase needs a proportionally larger carbon saving to justify it. But use
judgment based on the request's stated priority, not this as a strict rule
-- e.g. a "fastest" request should have a much higher bar than a "greenest"
one. A candidate whose carbon_delta_kg is zero or positive (no real
reduction) must always be rejected regardless of time_delta_s.

Respond with ONLY this JSON object:
    {"decisions": {"req-0": {"accept": true, "reasoning": "one sentence"}}}
"""


class CarbonOptimizerAgent:
    def __init__(self, costs_df: pd.DataFrame, model: str = None, penalized_edges: set = None):
        self.solver = RoutingSolver(costs_df)
        self.model = model
        # Edges the Coordinator (Phase 4) already worked to route around --
        # a carbon-motivated re-route must keep avoiding them too, or it
        # could silently reintroduce a conflict that was already resolved.
        self.penalized_edges = penalized_edges or set()

    def _call(self, system_prompt: str, payload: dict) -> dict:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload)},
        ]
        try:
            return chat_json(messages, model=self.model)
        except ValueError:
            return {}

    def optimize(
        self, results: list[RouteResult], requests_by_id: dict[str, DeliveryRequest]
    ) -> tuple[list[RouteResult], list[dict]]:
        """
        Returns (final_results, decisions_log). decisions_log has one entry
        per candidate actually evaluated (successful routes only), with the
        accept/reject outcome and the deltas it was based on -- useful for
        the Phase 8 eval harness and for auditing what the optimizer did.
        """
        results_by_id = {r.request_id: r for r in results}
        candidates_pool = [r for r in results_by_id.values() if r.success]
        if not candidates_pool:
            return list(results_by_id.values()), []

        propose_payload = {
            "routes": [
                {
                    "request_id": r.request_id,
                    "priority": r.priority,
                    "current_weights": r.weights_used,
                }
                for r in candidates_pool
            ]
        }
        proposal = self._call(PROPOSE_SYSTEM_PROMPT, propose_payload)
        candidate_weights = proposal.get("candidates", {})
        if not candidate_weights:
            return list(results_by_id.values()), []

        # GROUND: deterministically solve every candidate and compute deltas.
        candidate_results: dict[str, RouteResult] = {}
        deltas_by_id: dict[str, dict] = {}
        for r in candidates_pool:
            weights = candidate_weights.get(r.request_id)
            if weights is None:
                continue
            request = requests_by_id.get(r.request_id)
            if request is None:
                continue
            solver_result = self.solver.find_route(
                request.origin, request.destination,
                weights=weights, penalized_edges=self.penalized_edges,
            )
            if not solver_result["success"]:
                continue
            candidate_results[r.request_id] = build_route_result(request, solver_result, weights, None)
            deltas_by_id[r.request_id] = {
                "request_id": r.request_id,
                "priority": r.priority,
                "time_delta_s": solver_result["total_travel_time_s"] - r.total_travel_time_s,
                "carbon_delta_kg": solver_result["total_carbon_kg"] - r.total_carbon_kg,
                "original_time_s": r.total_travel_time_s,
                "original_carbon_kg": r.total_carbon_kg,
            }

        if not deltas_by_id:
            return list(results_by_id.values()), []

        # REFLECT: decide, per request, whether the ground-truth trade is
        # actually worth it.
        reflection = self._call(REFLECT_SYSTEM_PROMPT, {"candidates": list(deltas_by_id.values())})
        decisions = reflection.get("decisions", {})

        decisions_log = []
        for rid, delta in deltas_by_id.items():
            # Conservative default if the LLM call/parse failed or omitted
            # this request: reject, don't silently modify an already
            # conflict-free finalized plan on an unreviewed swap. Also
            # always reject a candidate that didn't even save carbon.
            decision = decisions.get(rid, {"accept": False, "reasoning": "no reflection response"})
            accept = bool(decision.get("accept")) and delta["carbon_delta_kg"] < 0

            if accept:
                results_by_id[rid] = candidate_results[rid]

            decisions_log.append({**delta, "accepted": accept, "reasoning": decision.get("reasoning")})

        return list(results_by_id.values()), decisions_log


# ══════════════════════════════════════════════════════════════════════════
# SMOKE TEST -- routes a small batch with an obviously time-flexible
# priority ("balanced"), then runs the optimizer over the result. Requires
# a working LLM key in .env.
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
    pairs = [(0, 2000), (100, 6000), (200, 7000)]
    priorities = ["balanced", "fastest", "greenest"]

    requests = [
        DeliveryRequest(request_id=f"req-{i}", origin=int(nodes[a]), destination=int(nodes[b]), priority=p)
        for i, ((a, b), p) in enumerate(zip(pairs, priorities))
    ]
    requests_by_id = {r.request_id: r for r in requests}

    route_agent = RouteAgent(costs_df)
    initial_results = [route_agent.route(r) for r in requests]

    print("\nBefore optimization:")
    for r in initial_results:
        print(f"  {r.request_id} priority={r.priority}  time={r.total_travel_time_s:.1f}s  carbon={r.total_carbon_kg:.3f}kg")

    optimizer = CarbonOptimizerAgent(costs_df)
    final_results, decisions_log = optimizer.optimize(initial_results, requests_by_id)

    print("\nAfter optimization:")
    final_by_id = {r.request_id: r for r in final_results}
    for entry in decisions_log:
        rid = entry["request_id"]
        r = final_by_id[rid]
        status = "ACCEPTED" if entry["accepted"] else "rejected"
        print(
            f"  {rid} [{status}] dt={entry['time_delta_s']:+.1f}s "
            f"dcarbon={entry['carbon_delta_kg']:+.3f}kg  reasoning: {entry['reasoning']}"
        )
        print(f"    final: time={r.total_travel_time_s:.1f}s carbon={r.total_carbon_kg:.3f}kg")
