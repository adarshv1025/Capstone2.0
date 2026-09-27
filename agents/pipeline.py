"""
pipeline.py

Phase 3: chains the Dispatcher (agents/dispatcher_agent.py) into parallel
Route Agents (agents/route_agent.py) across a batch of delivery requests.
Phase 4: `run_batch_with_negotiation` additionally runs the Coordinator
(agents/coordinator_agent.py) over that batch's output to resolve conflicts.
Phase 5: `run_batch_with_optimization` additionally runs the Carbon-Aware
Optimizer (agents/carbon_optimizer_agent.py) over the negotiated plan.
Phase 6: `run_batch_with_execution` additionally simulates execution of the
finalized plan via the Execution/Monitoring Agent
(agents/execution_agent.py).
Phase 7: `run_full_pipeline` additionally runs the ESG Reporter
(agents/esg_reporter_agent.py) over everything above -- the complete
6-agent roster, CLAUDE.md roster #1-6.

Route Agent calls are LLM round-trips (I/O-bound), so a thread pool is
enough to parallelize them -- no multiprocessing needed.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gnn"))

from carbon_optimizer_agent import CarbonOptimizerAgent
from coordinator_agent import CoordinatorAgent
from dispatcher_agent import DispatcherAgent
from esg_reporter_agent import ESGReporterAgent
from execution_agent import ExecutionAgent
from route_agent import RouteAgent, build_route_result
from schemas import DeliveryRequest, RouteResult, Vehicle


def _route_or_failure(route_agent: RouteAgent, request: DeliveryRequest) -> RouteResult:
    """
    Wraps RouteAgent.route() so one request's failure (e.g. a malformed LLM
    JSON response) can't abort the whole batch -- it becomes a failed
    RouteResult instead of propagating out of the thread pool. Resolving
    that failure is the Coordinator's job (Phase 4), not this layer's.
    """
    try:
        return route_agent.route(request)
    except Exception as e:
        return build_route_result(request, {"success": False, "reason": str(e)}, {}, None)


def run_batch(
    requests: list[DeliveryRequest],
    vehicles: list[Vehicle],
    costs_df: pd.DataFrame,
    model: str = None,
    max_workers: int = 4,
) -> tuple[list[RouteResult], list[DeliveryRequest]]:
    """
    Dispatches `requests` to `vehicles`, then routes every assigned request
    concurrently. Returns (route_results, unassigned_requests) -- the
    latter had no feasible vehicle and never reach the Route Agent.

    max_workers defaults to 4, not 8 (CLAUDE.md 4.5): testing found a batch
    of 12 requests at max_workers=8 consistently failed 4-5 of them to a
    per-minute rate limit, even on a fresh daily quota -- 8 Route Agent
    calls firing in the same instant is enough to blow through a
    rate-limited provider's TPM budget in one burst. A smaller burst size
    plus llm_client's retry jitter (see there) both help; this default
    trades a bit of parallelism for reliability under exactly the
    conditions (many concurrent LLM-dependent calls, a constrained API
    tier) a real deployment might also hit. Raise it back up for a
    generous/paid API tier where the burst isn't a concern.
    """
    dispatcher = DispatcherAgent(costs_df)
    assigned, unassigned = dispatcher.assign(requests, vehicles)

    route_agent = RouteAgent(costs_df, model=model)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(lambda r: _route_or_failure(route_agent, r), assigned))

    return results, unassigned


def run_batch_with_negotiation(
    requests: list[DeliveryRequest],
    vehicles: list[Vehicle],
    costs_df: pd.DataFrame,
    model: str = None,
    max_workers: int = 4,
    edge_capacity: int = 2,
    max_rounds: int = 3,
) -> tuple[list[RouteResult], list[DeliveryRequest], int, bool]:
    """
    Full Phase 4 flow: run_batch(), then CoordinatorAgent.resolve() over the
    result to clear shared-edge overload and EV-range conflicts. Returns
    (final_results, unassigned_requests, negotiation_rounds_used,
    fully_resolved). fully_resolved is False when max_rounds ran out with
    conflicts still outstanding -- see CoordinatorAgent.resolve's docstring
    for why that can be a genuinely infeasible request, not just a tuning
    failure.
    """
    results, unassigned = run_batch(requests, vehicles, costs_df, model=model, max_workers=max_workers)

    assigned_by_id = {r.request_id: r for r in requests if r.vehicle_id is not None}
    coordinator = CoordinatorAgent(
        costs_df, vehicles, model=model, edge_capacity=edge_capacity, max_rounds=max_rounds
    )
    final_results, rounds_used, fully_resolved = coordinator.resolve(results, assigned_by_id)

    return final_results, unassigned, rounds_used, fully_resolved


def run_batch_with_optimization(
    requests: list[DeliveryRequest],
    vehicles: list[Vehicle],
    costs_df: pd.DataFrame,
    model: str = None,
    max_workers: int = 4,
    edge_capacity: int = 2,
    max_rounds: int = 3,
) -> tuple[list[RouteResult], list[DeliveryRequest], int, bool, list[dict]]:
    """
    Full Phase 5 flow: run_batch_with_negotiation(), then
    CarbonOptimizerAgent.optimize() over the negotiated plan. Returns
    (final_results, unassigned_requests, negotiation_rounds_used,
    fully_resolved, carbon_decisions_log).

    Threads the Coordinator's penalized_edges through to the optimizer so a
    carbon-motivated re-route can't silently undo a conflict the
    Coordinator already resolved.
    """
    results, unassigned = run_batch(requests, vehicles, costs_df, model=model, max_workers=max_workers)

    assigned_by_id = {r.request_id: r for r in requests if r.vehicle_id is not None}
    coordinator = CoordinatorAgent(
        costs_df, vehicles, model=model, edge_capacity=edge_capacity, max_rounds=max_rounds
    )
    negotiated_results, rounds_used, fully_resolved = coordinator.resolve(results, assigned_by_id)

    optimizer = CarbonOptimizerAgent(costs_df, model=model, penalized_edges=coordinator.penalized_edges)
    final_results, decisions_log = optimizer.optimize(negotiated_results, assigned_by_id)

    return final_results, unassigned, rounds_used, fully_resolved, decisions_log


def run_batch_with_execution(
    requests: list[DeliveryRequest],
    vehicles: list[Vehicle],
    costs_df: pd.DataFrame,
    model: str = None,
    max_workers: int = 4,
    edge_capacity: int = 2,
    max_rounds: int = 3,
    seed: int = None,
) -> tuple[list[RouteResult], list[DeliveryRequest], int, bool, list[dict], list[dict]]:
    """
    Full Phase 6 flow: run_batch_with_optimization(), then
    ExecutionAgent.run() to simulate driving the finalized plan. Returns
    (final_results, unassigned_requests, negotiation_rounds_used,
    fully_resolved, carbon_decisions_log, execution_reports).

    seed is passed straight to ExecutionAgent for reproducible noise draws
    in tests; leave it None for a fresh simulation each call.
    """
    final_results, unassigned, rounds_used, fully_resolved, decisions_log = run_batch_with_optimization(
        requests, vehicles, costs_df, model=model, max_workers=max_workers,
        edge_capacity=edge_capacity, max_rounds=max_rounds,
    )

    vehicles_by_id = {v.vehicle_id: v for v in vehicles}
    executor = ExecutionAgent(costs_df, model=model, seed=seed)
    execution_reports = executor.run(final_results, vehicles_by_id)

    return final_results, unassigned, rounds_used, fully_resolved, decisions_log, execution_reports


def run_full_pipeline(
    requests: list[DeliveryRequest],
    vehicles: list[Vehicle],
    costs_df: pd.DataFrame,
    model: str = None,
    max_workers: int = 4,
    edge_capacity: int = 2,
    max_rounds: int = 3,
    seed: int = None,
) -> dict:
    """
    The complete pipeline (CLAUDE.md roster #1-6): Dispatcher -> parallel
    Route Agents -> Coordinator -> Carbon Optimizer -> Execution/Monitoring
    -> ESG Reporter.

    Returns a dict rather than continuing the growing positional tuple the
    run_batch_with_* helpers above use -- this is the final integration
    point and later work (the Phase 8 eval harness in particular) should
    reference these by name, not position:
        {
            "results": list[RouteResult],
            "unassigned": list[DeliveryRequest],
            "rounds_used": int,
            "fully_resolved": bool,
            "decisions_log": list[dict],       # Carbon Optimizer
            "execution_reports": list[dict],   # Execution/Monitoring
            "esg_report": {"metrics": {...}, "summary": str},
        }
    """
    results, unassigned, rounds_used, fully_resolved, decisions_log, execution_reports = run_batch_with_execution(
        requests, vehicles, costs_df, model=model, max_workers=max_workers,
        edge_capacity=edge_capacity, max_rounds=max_rounds, seed=seed,
    )

    vehicles_by_id = {v.vehicle_id: v for v in vehicles}
    reporter = ESGReporterAgent(model=model)
    esg_report = reporter.report(results, unassigned, vehicles_by_id, decisions_log, execution_reports, rounds_used)

    return {
        "results": results,
        "unassigned": unassigned,
        "rounds_used": rounds_used,
        "fully_resolved": fully_resolved,
        "decisions_log": decisions_log,
        "execution_reports": execution_reports,
        "esg_report": esg_report,
    }


# ══════════════════════════════════════════════════════════════════════════
# SMOKE TEST -- runs the complete pipeline (Dispatcher -> parallel Route
# Agents -> Coordinator -> Carbon Optimizer -> Execution/Monitoring -> ESG
# Reporter) against sample_costs/*.csv. Requires a working LLM key in .env.
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

    nodes = pd.unique(costs_df[["src", "dst"]].values.ravel())
    priorities = ["fastest", "greenest", "balanced", "fastest", "balanced"]

    requests = [
        DeliveryRequest(
            request_id=f"req-{i}",
            origin=int(nodes[i * 7]),
            destination=int(nodes[-(i + 1)]),
            priority=priorities[i % len(priorities)],
            package_weight_kg=50.0,
        )
        for i in range(5)
    ]
    vehicles = [
        Vehicle(
            vehicle_id=f"veh-{i}",
            is_ev=(i % 2 == 0),
            battery_capacity_kwh=50.0,
            max_payload_kg=200.0,
            current_location=int(nodes[i * 3]),
        )
        for i in range(5)
    ]

    pipeline_result = run_full_pipeline(requests, vehicles, costs_df, edge_capacity=2)
    results = pipeline_result["results"]
    unassigned = pipeline_result["unassigned"]
    rounds_used = pipeline_result["rounds_used"]
    fully_resolved = pipeline_result["fully_resolved"]
    decisions_log = pipeline_result["decisions_log"]
    execution_reports = pipeline_result["execution_reports"]
    esg_report = pipeline_result["esg_report"]

    print(
        f"\n{len(results)} routed, {len(unassigned)} unassigned, "
        f"{rounds_used} negotiation round(s) used, fully_resolved={fully_resolved}\n"
    )
    for r in results:
        status = "OK" if r.success else f"FAILED ({r.reason})"
        print(
            f"[{r.request_id} -> {r.vehicle_id}] {status} priority={r.priority} "
            f"weights={r.weights_used}"
        )
        if r.success:
            print(
                f"    time={r.total_travel_time_s:.1f}s carbon={r.total_carbon_kg:.3f}kg "
                f"delay_p={r.route_delay_probability:.3f}"
            )
        if r.reasoning:
            print(f"    reasoning: {r.reasoning}")

    print(f"\nCarbon Optimizer decisions ({len(decisions_log)} candidate(s) evaluated):")
    for entry in decisions_log:
        status = "ACCEPTED" if entry["accepted"] else "rejected"
        print(
            f"  {entry['request_id']} [{status}] dt={entry['time_delta_s']:+.1f}s "
            f"dcarbon={entry['carbon_delta_kg']:+.3f}kg  reasoning: {entry['reasoning']}"
        )

    print(f"\nExecution simulation ({len(execution_reports)} route(s) simulated):")
    for rep in execution_reports:
        flag = "REPLAN" if rep["replan_triggered"] else "ok"
        print(
            f"  {rep['request_id']} [{flag}] planned={rep['planned_time_s']:.1f}s "
            f"actual={rep['actual_time_s']:.1f}s (dev={rep['time_deviation_pct']:+.1%}) "
            f"delayed_edges={rep['num_delayed_edges']}/{rep['num_edges']}"
        )
        if rep["reasoning"]:
            print(f"      reasoning: {rep['reasoning']}")

    print("\nESG Report metrics:")
    for k, v in esg_report["metrics"].items():
        print(f"  {k}: {v}")
    print(f"\nESG Report summary:\n  {esg_report['summary']}")
