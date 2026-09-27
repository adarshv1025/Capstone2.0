"""
esg_reporter_agent.py

Phase 7: ESG Reporter (CLAUDE.md roster #6) -- aggregates + narrates
sustainability metrics for the dashboard. The one agent in this roster that
summarizes across the WHOLE run rather than deciding per-route, so there's
no per-item-call scaling concern like Phases 4-6 hit -- this is naturally
one LLM call per run regardless of batch size.

Deterministic grounding: every number in the report is computed in code
from the pipeline's own outputs (the RouteResult list, the Carbon
Optimizer's decisions_log, the Execution Agent's execution_reports, the
Coordinator's negotiation round count) -- nothing here is invented or
estimated. LLM judgment: narrates those numbers into a short human-readable
summary. Per CLAUDE.md 3.2, the LLM narrates facts computed in code, it
does not calculate them -- the narrate prompt is told every number is
ground truth and not to invent its own.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gnn"))

from llm_client import chat_json
from schemas import DeliveryRequest, RouteResult, Vehicle

NARRATE_SYSTEM_PROMPT = """You are the ESG Reporter for a fleet delivery platform. You are given a
JSON object of aggregate metrics from one completed delivery run -- every
number in it was computed deterministically from the run's own results, so
treat them all as ground truth. Never invent or estimate a fact that isn't
in the JSON -- but DO round numbers for readability when you write them
(e.g. write "15,672s (~4.4 hours)" or "35.1 kg", not
"15671.69233300773 s"); rounding for display is not inventing a number, as
long as it doesn't change what the underlying fact says.

Notes on specific fields: `negotiation_rounds_used` is a total for the
WHOLE batch/run, not a per-request count -- phrase it that way (e.g. "the
batch needed 3 negotiation rounds to resolve conflicts"), not "3 rounds per
request". `mean_abs_time_deviation_pct` is the average, across routes, of
how far off (in either direction) each route's actual travel time was from
its planned time -- treat it as the primary on-time-performance figure and
`on_time_rate` as a secondary, coarser one (it can only take a few discrete
values on a small batch, so don't over-read small differences in it).

Write a short sustainability summary (3-5 sentences) for a dashboard, aimed
at a non-technical stakeholder. Highlight what's notable: strong
performance worth celebrating, and any weak spots worth flagging (e.g. a
low on-time rate, EV infeasibility, negotiation friction, requests that
never got assigned a vehicle). Reference the actual numbers given rather
than describing them vaguely.

Respond with ONLY this JSON object:
    {"summary": "your 3-5 sentence narrative here"}
"""


class ESGReporterAgent:
    def __init__(self, model: str = None):
        self.model = model

    def aggregate(
        self,
        results: list[RouteResult],
        unassigned: list[DeliveryRequest],
        vehicles_by_id: dict[str, Vehicle],
        decisions_log: list[dict],
        execution_reports: list[dict],
        negotiation_rounds_used: int,
    ) -> dict:
        """
        Deterministic grounding -- every number here is computed from the
        pipeline's own outputs, nothing estimated.
        """
        successful = [r for r in results if r.success]
        total_requests = len(results) + len(unassigned)
        num_unassigned = len(unassigned)
        num_routing_failed = len(results) - len(successful)

        total_planned_time_s = sum(r.total_travel_time_s for r in successful)
        total_planned_carbon_kg = sum(r.total_carbon_kg for r in successful)

        reports_by_id = {rep["request_id"]: rep for rep in execution_reports}
        simulated = [reports_by_id[r.request_id] for r in successful if r.request_id in reports_by_id]
        total_actual_time_s = sum(rep["actual_time_s"] for rep in simulated)
        total_actual_carbon_kg = sum(rep["actual_carbon_kg"] for rep in simulated)

        num_replan_triggered = sum(1 for rep in simulated if rep["replan_triggered"])
        on_time_rate = (len(simulated) - num_replan_triggered) / len(simulated) if simulated else None
        # Continuous companion to on_time_rate (CLAUDE.md 4.2 item 2): with
        # a small batch, on_time_rate can only take a handful of discrete
        # values (e.g. 6 for a 5-request batch), so two systems with
        # genuinely different routes can land on the identical rate by
        # coincidence (observed in Phase 8's first evaluation run, all 3
        # systems tied at 0.20 on mixed_priority). Mean absolute deviation
        # doesn't collapse like that -- it's the primary comparison metric
        # for on-time performance; on_time_rate is kept as a secondary,
        # more intuitive stat.
        mean_abs_time_deviation_pct = (
            sum(abs(rep["time_deviation_pct"]) for rep in simulated) / len(simulated) if simulated else None
        )

        ev_reports = [rep for rep in simulated if (vehicles_by_id.get(rep["vehicle_id"]) or Vehicle(vehicle_id="_")).is_ev]
        num_ev_infeasible = sum(1 for rep in ev_reports if rep["hard_ev_violation"])
        ev_feasibility_rate = (len(ev_reports) - num_ev_infeasible) / len(ev_reports) if ev_reports else None

        accepted_swaps = [d for d in decisions_log if d["accepted"]]
        # accepted swaps always have carbon_delta_kg < 0 (Phase 5's hard
        # rule), so negating the sum turns it into a positive "kg saved".
        carbon_saved_by_optimizer_kg = -sum(d["carbon_delta_kg"] for d in accepted_swaps)

        return {
            "total_requests": total_requests,
            "num_routed_successfully": len(successful),
            "num_unassigned": num_unassigned,
            "num_routing_failed": num_routing_failed,
            "total_planned_time_s": total_planned_time_s,
            "total_actual_time_s": total_actual_time_s,
            "total_planned_carbon_kg": total_planned_carbon_kg,
            "total_actual_carbon_kg": total_actual_carbon_kg,
            "on_time_rate": on_time_rate,
            "mean_abs_time_deviation_pct": mean_abs_time_deviation_pct,
            "num_replan_triggered": num_replan_triggered,
            "ev_feasibility_rate": ev_feasibility_rate,
            "num_ev_infeasible": num_ev_infeasible,
            "carbon_optimizer_swaps_accepted": len(accepted_swaps),
            "carbon_optimizer_swaps_evaluated": len(decisions_log),
            "carbon_saved_by_optimizer_kg": carbon_saved_by_optimizer_kg,
            "negotiation_rounds_used": negotiation_rounds_used,
        }

    def narrate(self, metrics: dict) -> str:
        messages = [
            {"role": "system", "content": NARRATE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(metrics)},
        ]
        try:
            result = chat_json(messages, model=self.model)
        except ValueError:
            return "(narrative unavailable -- LLM did not return valid JSON)"
        return result.get("summary", "(narrative unavailable)")

    def report(
        self,
        results: list[RouteResult],
        unassigned: list[DeliveryRequest],
        vehicles_by_id: dict[str, Vehicle],
        decisions_log: list[dict],
        execution_reports: list[dict],
        negotiation_rounds_used: int,
    ) -> dict:
        """
        Full Phase 7 flow: aggregate metrics deterministically, then have
        the LLM narrate them. Returns {"metrics": {...}, "summary": "..."}.
        """
        metrics = self.aggregate(
            results, unassigned, vehicles_by_id, decisions_log, execution_reports, negotiation_rounds_used
        )
        summary = self.narrate(metrics)
        return {"metrics": metrics, "summary": summary}


# ══════════════════════════════════════════════════════════════════════════
# SMOKE TEST -- runs the full pipeline then reports on it. Requires a
# working LLM key in .env.
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import glob

    # LLM output can contain Unicode punctuation (e.g. non-breaking hyphens)
    # that Windows' default console codepage can't print -- force utf-8.
    sys.stdout.reconfigure(encoding="utf-8")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pandas as pd

    from pipeline import run_batch_with_execution
    from schemas import DeliveryRequest, Vehicle

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

    results, unassigned, rounds_used, fully_resolved, decisions_log, execution_reports = run_batch_with_execution(
        requests, vehicles, costs_df, edge_capacity=2
    )
    vehicles_by_id = {v.vehicle_id: v for v in vehicles}

    reporter = ESGReporterAgent()
    esg_report = reporter.report(
        results, unassigned, vehicles_by_id, decisions_log, execution_reports, rounds_used
    )

    print("\nMetrics:")
    for k, v in esg_report["metrics"].items():
        print(f"  {k}: {v}")

    print("\nSummary:")
    print(f"  {esg_report['summary']}")
