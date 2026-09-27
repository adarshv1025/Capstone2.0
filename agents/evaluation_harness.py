"""
evaluation_harness.py

Phase 8 (final phase): the evaluation harness (CLAUDE.md 3.5 phase 8 / 3.12).
This directly serves the project's stated goal (CLAUDE.md section 1):
whichever system -- this one, or a teammate's independent implementation of
the same layers -- scores better here is what ships in the final
submission. So this file, more than any other in agents/, has to be fair
and trustworthy, not flattering to the system that happens to be built here.

Runs the full multi-agent pipeline (pipeline.run_full_pipeline) AND two
deterministic baselines against the SAME golden scenarios
(golden_scenarios.GOLDEN_SCENARIOS) and the SAME costs_df snapshot, scored
through the SAME metrics (ESGReporterAgent.aggregate) -- an apples-to-apples
comparison, not three different measurement methodologies.

APPLES-TO-APPLES DESIGN NOTE: a "baseline" here means no LLM anywhere in the
ROUTING decision -- RoutingSolver called directly with fixed weights, no
Route Agent, no Coordinator negotiation, no Carbon Optimizer swaps. But
baselines still go through the SAME DispatcherAgent (assignment quality
isn't what's being compared) and the SAME ExecutionAgent simulation +
ESGReporterAgent metrics as the multi-agent system, including
ExecutionAgent's own LLM-judged soft-deviation reflection. That's
deliberate: on_time_rate and the other metrics need to mean the same thing
for every system being compared, so the measurement machinery has to be
identical -- only the routing/negotiation/optimization decisions differ.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gnn"))
from routing_solver import RoutingSolver  # noqa: E402

from dispatcher_agent import DispatcherAgent
from esg_reporter_agent import ESGReporterAgent
from execution_agent import ExecutionAgent
from golden_scenarios import GOLDEN_SCENARIOS
from pipeline import run_full_pipeline
from route_agent import build_route_result
from schemas import DeliveryRequest, RouteResult, Vehicle

BASELINE_WEIGHTS = {
    "baseline_time_only": {"time": 1.0, "delay": 0.0, "carbon": 0.0, "ev": 0.0},
    "baseline_carbon_only": {"time": 0.0, "delay": 0.0, "carbon": 1.0, "ev": 0.0},
}


def run_baseline(
    requests: list[DeliveryRequest],
    vehicles: list[Vehicle],
    costs_df: pd.DataFrame,
    weights: dict,
    model: str = None,
    seed: int = None,
) -> dict:
    """
    A deterministic baseline "system": fixed-weight RoutingSolver calls, no
    LLM anywhere in the routing decision. Still uses the same
    DispatcherAgent / ExecutionAgent / ESGReporterAgent as
    run_full_pipeline() -- see the module docstring for why. Returns the
    same shape run_full_pipeline() does, with rounds_used=0 and
    decisions_log=[] since baselines never negotiate or optimize.
    """
    dispatcher = DispatcherAgent(costs_df)
    assigned, unassigned = dispatcher.assign(requests, vehicles)

    solver = RoutingSolver(costs_df)
    results: list[RouteResult] = []
    for request in assigned:
        solver_result = solver.find_route(request.origin, request.destination, weights=weights)
        results.append(build_route_result(request, solver_result, weights, None))

    vehicles_by_id = {v.vehicle_id: v for v in vehicles}
    executor = ExecutionAgent(costs_df, model=model, seed=seed)
    execution_reports = executor.run(results, vehicles_by_id)

    reporter = ESGReporterAgent(model=model)
    metrics = reporter.aggregate(results, unassigned, vehicles_by_id, [], execution_reports, 0)

    return {
        "results": results,
        "unassigned": unassigned,
        "rounds_used": 0,
        "fully_resolved": None,  # not applicable -- baselines never attempt conflict resolution
        "decisions_log": [],
        "execution_reports": execution_reports,
        "esg_report": {"metrics": metrics, "summary": None},  # narration skipped -- harness only needs numbers
    }


def _run_multi_agent(requests, vehicles, costs_df, model, seed):
    return run_full_pipeline(requests, vehicles, costs_df, model=model, seed=seed)


SYSTEMS = {
    "multi_agent": _run_multi_agent,
    "baseline_time_only": lambda requests, vehicles, costs_df, model, seed: run_baseline(
        requests, vehicles, costs_df, BASELINE_WEIGHTS["baseline_time_only"], model=model, seed=seed
    ),
    "baseline_carbon_only": lambda requests, vehicles, costs_df, model, seed: run_baseline(
        requests, vehicles, costs_df, BASELINE_WEIGHTS["baseline_carbon_only"], model=model, seed=seed
    ),
}


def run_evaluation(
    costs_df: pd.DataFrame, model: str = None, seed: int = None, scenarios=GOLDEN_SCENARIOS
) -> pd.DataFrame:
    """
    Runs every system in SYSTEMS against every scenario in `scenarios`,
    all against the same costs_df. Returns a DataFrame with one row per
    (scenario, system) and every metric ESGReporterAgent.aggregate()
    computes as columns, plus a `reliable` column (see below).

    IMPORTANT CAVEAT, found via testing (don't silently drop this column):
    `total_planned_time_s` / `total_planned_carbon_kg` / etc. are SUMS over
    only the successfully-routed requests (`num_routed_successfully`). If a
    system fails to route some requests (e.g. a Route Agent call exhausting
    its retries under a tight LLM provider rate limit -- observed on the
    `large_batch` scenario once the day's token budget ran low) while
    another system routes all of them, comparing their totals directly is
    NOT apples-to-apples: fewer successfully-routed requests can make a
    system look faster/greener when it actually just did less work. The
    `reliable` column (True iff `num_routing_failed == 0`) flags this --
    treat a row with `reliable=False` as informative about robustness under
    real-world constraints, not as a valid basis for a totals comparison
    against a `reliable=True` row for the same scenario.
    """
    rows = []
    for scenario_name, builder in scenarios:
        for system_name, run_fn in SYSTEMS.items():
            # Fresh requests/vehicles per system: DispatcherAgent.assign()
            # mutates request.vehicle_id in place, so reusing one set of
            # objects across systems would leak state between them.
            requests, vehicles = builder(costs_df)
            result = run_fn(requests, vehicles, costs_df, model, seed)
            metrics = result["esg_report"]["metrics"]
            rows.append({
                "scenario": scenario_name,
                "system": system_name,
                "fully_resolved": result["fully_resolved"],
                "reliable": metrics["num_routing_failed"] == 0,
                **metrics,
            })

    return pd.DataFrame(rows)


def summarize_across_snapshots(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates a raw multi-snapshot results DataFrame (one row per
    (snapshot, scenario, system), e.g. from run_multi_snapshot_evaluation or
    from incrementally concatenating run_evaluation() calls) into one row
    per (scenario, system), with each numeric metric's mean and std across
    snapshots as separate columns (`<metric>_mean`, `<metric>_std`) -- std
    is NaN for a metric that's constant across snapshots or when only one
    snapshot is present, not an error.
    """
    # bool columns (e.g. `reliable`) aren't picked up by select_dtypes's
    # "number" filter in this pandas version -- cast them to int first so
    # e.g. reliable_mean (fraction of snapshots with zero routing failures)
    # comes through instead of silently disappearing from the summary.
    raw_df = raw_df.assign(**{c: raw_df[c].astype(int) for c in raw_df.select_dtypes(include="bool").columns})
    numeric_cols = raw_df.select_dtypes(include="number").columns.tolist()
    summary_df = raw_df.groupby(["scenario", "system"])[numeric_cols].agg(["mean", "std"])
    summary_df.columns = [f"{col}_{stat}" for col, stat in summary_df.columns]
    return summary_df.reset_index()


def run_multi_snapshot_evaluation(
    snapshot_paths: list[str], model: str = None, seed: int = None, scenarios=GOLDEN_SCENARIOS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    CLAUDE.md 4.2 item 1: runs run_evaluation() once per snapshot in
    `snapshot_paths` (checking whether Phase 8's single-snapshot findings
    hold across different traffic conditions or were specific to that one
    snapshot), then aggregates via summarize_across_snapshots(). Returns
    (raw_df, summary_df) -- raw_df has one row per (snapshot, scenario,
    system), every individual run's metrics, unaggregated.

    For a long run (many snapshots x many scenarios), prefer looping over
    run_evaluation() directly and saving raw_df incrementally after each
    snapshot (see run_full_evaluation.py) -- this function only returns
    once everything has finished, so an interruption partway through loses
    all of it.
    """
    raw_rows = []
    for path in snapshot_paths:
        costs_df = pd.read_csv(path)
        df = run_evaluation(costs_df, model=model, seed=seed, scenarios=scenarios)
        df.insert(0, "snapshot", Path(path).stem)
        raw_rows.append(df)

    raw_df = pd.concat(raw_rows, ignore_index=True)
    return raw_df, summarize_across_snapshots(raw_df)


# ══════════════════════════════════════════════════════════════════════════
# SMOKE TEST -- runs all 3 systems against all 3 golden scenarios (9 full
# runs) against one sample_costs snapshot. Requires a working LLM key in
# .env. seed=0 makes the Execution Agent's noise draws reproducible.
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
    print(f"Loaded {csv_files[0]} ({len(costs_df)} edges)\n")

    results_df = run_evaluation(costs_df, seed=0)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    cols = [
        "scenario", "system", "num_routed_successfully", "num_unassigned",
        "total_planned_time_s", "total_actual_time_s", "total_planned_carbon_kg",
        "total_actual_carbon_kg", "on_time_rate", "ev_feasibility_rate",
        "negotiation_rounds_used", "fully_resolved",
    ]
    print(results_df[cols].to_string(index=False))
