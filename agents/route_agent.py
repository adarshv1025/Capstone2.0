"""
route_agent.py

Phase 2: a single Route Agent, wired up end-to-end (CLAUDE.md section 3.6).

Takes a DeliveryRequest with a fuzzy priority ("fastest", "greenest",
"balanced", or free text), has the LLM translate that priority into a
{time, delay, carbon, ev} weights dict, calls the deterministic
RoutingSolver with those weights, and returns a structured RouteResult.

Design principle (CLAUDE.md 3.2): the deterministic tool (RoutingSolver)
does the grounding; the LLM does the judgment (interpreting the priority
into weights). The LLM never touches the graph or the routing math directly.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gnn"))
from routing_solver import RoutingSolver  # noqa: E402

from llm_client import chat_json
from schemas import DeliveryRequest, RouteResult

def build_route_result(
    request: DeliveryRequest, solver_result: dict, weights: dict, reasoning: str | None
) -> RouteResult:
    """
    Turns a RoutingSolver.find_route() result into a RouteResult. Shared by
    RouteAgent.route() and CoordinatorAgent (agents/coordinator_agent.py,
    Phase 4), which re-solves individual routes during conflict resolution
    and needs the exact same construction logic.
    """
    if not solver_result["success"]:
        return RouteResult(
            request_id=request.request_id,
            vehicle_id=request.vehicle_id,
            success=False,
            reason=solver_result["reason"],
            priority=request.priority,
            weights_used=weights,
            reasoning=reasoning,
        )

    return RouteResult(
        request_id=request.request_id,
        vehicle_id=request.vehicle_id,
        success=True,
        path=solver_result["path"],
        num_edges=solver_result["num_edges"],
        total_travel_time_s=solver_result["total_travel_time_s"],
        route_delay_probability=solver_result["route_delay_probability"],
        total_carbon_kg=solver_result["total_carbon_kg"],
        total_ev_energy_pct=solver_result["total_ev_energy_pct"],
        priority=request.priority,
        weights_used=weights,
        reasoning=reasoning,
    )


SYSTEM_PROMPT = """You are a routing weights translator for a logistics platform.

Given a delivery's stated priority, output a JSON object with exactly these
four keys, each a non-negative float, expressing the RELATIVE importance of
each objective for this delivery's route (they will be renormalized to sum
to 1, so any consistent scale is fine):

    "time"   -- minimize total travel time
    "delay"  -- minimize probability of a significant delay
    "carbon" -- minimize CO2 emitted
    "ev"     -- minimize EV battery energy consumed

Also include a "reasoning" key: one short sentence explaining your choice.

Respond with ONLY the JSON object, no other text. Example:
{"time": 0.6, "delay": 0.2, "carbon": 0.1, "ev": 0.1, "reasoning": "..."}
"""


class RouteAgent:
    def __init__(self, costs_df: pd.DataFrame, model: str = None):
        self.solver = RoutingSolver(costs_df)
        self.model = model

    def _weights_from_priority(self, priority: str) -> tuple[dict, str | None]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Delivery priority: {priority!r}"},
        ]
        result = chat_json(messages, model=self.model)
        weights = {
            "time": float(result.get("time", 0.25)),
            "delay": float(result.get("delay", 0.25)),
            "carbon": float(result.get("carbon", 0.25)),
            "ev": float(result.get("ev", 0.25)),
        }
        return weights, result.get("reasoning")

    def route(self, request: DeliveryRequest) -> RouteResult:
        weights, reasoning = self._weights_from_priority(request.priority)
        result = self.solver.find_route(request.origin, request.destination, weights=weights)
        return build_route_result(request, result, weights, reasoning)


# ══════════════════════════════════════════════════════════════════════════
# SMOKE TEST -- runs against sample_costs/*.csv, no torch/GNN checkpoint
# needed (CLAUDE.md 2.6 / 3.6). Requires an API key set in .env.
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import glob

    # LLM output can contain Unicode punctuation (e.g. non-breaking hyphens)
    # that Windows' default console codepage can't print -- force utf-8.
    sys.stdout.reconfigure(encoding="utf-8")

    sample_costs_dir = Path(__file__).resolve().parent.parent / "sample_costs"
    csv_files = sorted(glob.glob(str(sample_costs_dir / "*.csv")))
    if not csv_files:
        raise SystemExit(f"No sample cost CSVs found in {sample_costs_dir}")

    costs_df = pd.read_csv(csv_files[0])
    print(f"Loaded {csv_files[0]} ({len(costs_df)} edges)")

    agent = RouteAgent(costs_df)

    origin = int(costs_df.iloc[0]["src"])
    destination = int(costs_df.iloc[-1]["dst"])

    for priority in ["fastest", "greenest", "balanced"]:
        request = DeliveryRequest(
            request_id=f"test-{priority}",
            origin=origin,
            destination=destination,
            priority=priority,
        )
        result = agent.route(request)
        print(f"\n[{priority}] -> weights={result.weights_used}")
        print(f"  reasoning: {result.reasoning}")
        if result.success:
            print(
                f"  time={result.total_travel_time_s:.1f}s "
                f"carbon={result.total_carbon_kg:.3f}kg "
                f"delay_p={result.route_delay_probability:.3f} "
                f"ev={result.total_ev_energy_pct:.2f}%"
            )
        else:
            print(f"  FAILED: {result.reason}")
