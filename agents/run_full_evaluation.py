"""
run_full_evaluation.py

CLAUDE.md 4.2 item 1: runs the full evaluation across all 5 sample_costs
snapshots and all golden scenarios/systems, then reports mean +/- std per
(scenario, system) -- checking whether Phase 8's single-snapshot findings
hold across different traffic conditions, or were specific to that one
snapshot.

This makes ~75 LLM-backed full-pipeline calls in total (5 scenarios x 3
systems x 5 snapshots) and can take a long time even with llm_client's
built-in rate-limit retry -- long enough to risk an interruption (a
rate-limited provider outage, a timeout on whatever is running this). So
this script supports running ONE snapshot at a time and merging into a
shared CSV, not just an all-or-nothing batch:

    python run_full_evaluation.py            # all snapshots in one go
    python run_full_evaluation.py 0          # just snapshot index 0
    python run_full_evaluation.py <path.csv> # just that snapshot file
    python run_full_evaluation.py --summarize  # recompute summary from
                                                # whatever's in raw_results.csv

Each single-snapshot run merges its rows into evaluation_results/raw_results.csv,
replacing only that snapshot's prior rows (safe to re-run) -- so 5 separate
invocations (one per snapshot) produce the exact same raw_results.csv as one
big run, but each invocation is short enough to comfortably avoid a
long-running-process timeout, and a failure partway through only costs that
one snapshot's work, not everything done so far.

Requires a working LLM key in .env.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation_harness import run_evaluation, summarize_across_snapshots

if __name__ == "__main__":
    import glob

    # LLM output can contain Unicode punctuation (e.g. non-breaking hyphens)
    # that Windows' default console codepage can't print -- force utf-8.
    sys.stdout.reconfigure(encoding="utf-8")

    sample_costs_dir = Path(__file__).resolve().parent.parent / "sample_costs"
    all_snapshot_paths = sorted(glob.glob(str(sample_costs_dir / "*.csv")))
    if not all_snapshot_paths:
        raise SystemExit(f"No sample cost CSVs found in {sample_costs_dir}")

    out_dir = Path(__file__).resolve().parent / "evaluation_results"
    out_dir.mkdir(exist_ok=True)
    raw_path = out_dir / "raw_results.csv"
    summary_path = out_dir / "summary_results.csv"

    arg = sys.argv[1] if len(sys.argv) > 1 else None
    summary_cols = [
        "scenario", "system",
        "total_planned_time_s_mean", "total_planned_time_s_std",
        "total_planned_carbon_kg_mean", "total_planned_carbon_kg_std",
        "mean_abs_time_deviation_pct_mean", "mean_abs_time_deviation_pct_std",
        "on_time_rate_mean", "on_time_rate_std",
        "ev_feasibility_rate_mean", "ev_feasibility_rate_std",
        "negotiation_rounds_used_mean", "negotiation_rounds_used_std",
    ]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)

    if arg == "--summarize":
        if not raw_path.exists():
            raise SystemExit(f"{raw_path} doesn't exist yet -- run at least one snapshot first.")
        raw_df = pd.read_csv(raw_path)
        summary_df = summarize_across_snapshots(raw_df)
        summary_df.to_csv(summary_path, index=False)
        print(f"Summarized {len(raw_df)} raw rows ({raw_df['snapshot'].nunique()} snapshot(s)) -> {summary_path}")
        print("\nSummary (mean +/- std across snapshots):")
        print(summary_df[summary_cols].to_string(index=False))
        raise SystemExit(0)

    if arg is None:
        snapshot_paths = all_snapshot_paths
    elif arg.isdigit():
        snapshot_paths = [all_snapshot_paths[int(arg)]]
    else:
        snapshot_paths = [arg]

    print(f"Running evaluation for {len(snapshot_paths)} snapshot(s):", flush=True)
    for p in snapshot_paths:
        print(f"  {p}", flush=True)

    new_dfs = []
    for i, path in enumerate(snapshot_paths, 1):
        print(f"\n[{i}/{len(snapshot_paths)}] {Path(path).name} ...", flush=True)
        costs_df = pd.read_csv(path)
        df = run_evaluation(costs_df, seed=0)
        df.insert(0, "snapshot", Path(path).stem)
        new_dfs.append(df)
        print(f"  done ({len(df)} rows)", flush=True)

    new_df = pd.concat(new_dfs, ignore_index=True)

    # Merge into any existing raw CSV, replacing only the snapshot(s) just
    # (re)computed -- lets this be called once per snapshot across several
    # separate invocations without duplicating or losing prior rows.
    if raw_path.exists():
        existing_df = pd.read_csv(raw_path)
        existing_df = existing_df[~existing_df["snapshot"].isin(new_df["snapshot"].unique())]
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_csv(raw_path, index=False)
    print(f"\nSaved {len(combined_df)} total rows ({combined_df['snapshot'].nunique()} snapshot(s)) to {raw_path}")

    if combined_df["snapshot"].nunique() >= len(all_snapshot_paths):
        summary_df = summarize_across_snapshots(combined_df)
        summary_df.to_csv(summary_path, index=False)
        print(f"All {len(all_snapshot_paths)} snapshots present -- saved summary to {summary_path}")
        print("\nSummary (mean +/- std across snapshots):")
        print(summary_df[summary_cols].to_string(index=False))
    else:
        print("Run again with the remaining snapshot(s), or with --summarize once all are done.")
