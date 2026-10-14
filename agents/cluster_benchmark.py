"""
cluster_benchmark.py

Measures the distributed runtime (agents/cluster.py) against a plain serial
run, and checks that both give the same answers.

Jobs:
  gnn     GNN inference over every test snapshot (381), combined into
          pooled R^2 per target. Serial: one process, PyTorch's default
          threads. Distributed: N worker processes, 12 // N threads each.
  replan  The closed-loop experiment (agents/replan_experiment.py) on
          --trips trips; serial against N workers, one trip per task.

Times include each worker loading the GNN (and, for replan, building its
hour's worlds) -- the real cost of starting a run.

    python agents/cluster_benchmark.py                    # workers 1 2 4, both jobs
    python agents/cluster_benchmark.py --jobs replan --workers 4 --trips 48
    python agents/cluster_benchmark.py --scheduler tcp://host:8786   # an existing cluster

Writes agents/evaluation_results/cluster_benchmark.csv.
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agents"))
sys.path.insert(0, str(ROOT / "gnn"))
warnings.filterwarnings("ignore", category=FutureWarning)

import cluster  # noqa: E402

OUT = ROOT / "agents" / "evaluation_results" / "cluster_benchmark.csv"
PHYSICAL_CORES = 12   # this laptop (i5-13500H); used only to split torch threads between workers


def snapshot_names() -> list[str]:
    index = json.loads((ROOT / "dataset_kaggle" / "test" / "_index.json").read_text())
    return sorted(f"{name}.pt" for name in index)


def gnn_serial(names):
    return [cluster.score_snapshot(n) for n in names]


def compare_replan(a: tuple, b: tuple) -> bool:
    (ra, fa), (rb, fb) = a, b
    if len(ra) != len(rb) or len(fa) != len(fb):
        return False
    key = ["trip", "kind"]
    ra, rb = ra.sort_values(key).reset_index(drop=True), rb.sort_values(key).reset_index(drop=True)
    if not (ra["replan"].equals(rb["replan"]) and ra["rule"].equals(rb["rule"])):
        return False
    numeric = ["time_with_s", "time_without_s", "saved_time_s", "carbon_with_kg"]
    return bool(((ra[numeric] - rb[numeric]).abs() <= 1e-6 * (1 + ra[numeric].abs())).all().all())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--jobs", nargs="+", default=["gnn", "replan"], choices=["gnn", "replan"])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--trips", type=int, default=24)
    parser.add_argument("--scheduler", default=None)
    parser.add_argument("--out", default=str(OUT), help="CSV to write")
    args = parser.parse_args()

    rows = []

    if "gnn" in args.jobs:
        names = snapshot_names()
        started = time.perf_counter()
        serial = gnn_serial(names)
        serial_s = time.perf_counter() - started
        serial_r2 = cluster.combine_r2(serial)
        print(f"gnn serial: {len(names)} snapshots in {serial_s:.1f}s  R2 {', '.join(f'{v:.4f}' for v in serial_r2.values())}")
        rows.append({"job": "gnn", "mode": "serial", "workers": 1, "seconds": serial_s, "items": len(names),
                     "speedup": 1.0, "same_answer": True, "worker_processes_seen": 1,
                     **{f"r2_{k}": v for k, v in serial_r2.items()}})
        for n in ([None] if args.scheduler else args.workers):
            client = cluster.connect(args.scheduler, workers=n or 2)
            try:
                started = time.perf_counter()
                parts = cluster.score_snapshots(client, names, torch_threads=max(1, PHYSICAL_CORES // (n or 2)))
                seconds = time.perf_counter() - started
            finally:
                cluster.release(client)
            r2 = cluster.combine_r2(parts)
            same = all(abs(r2[k] - serial_r2[k]) < 1e-9 for k in r2)
            seen = len({p["worker"] for p in parts})
            print(f"gnn {n or 'cluster'} worker(s): {seconds:.1f}s  speedup x{serial_s / seconds:.2f}  same R2: {same}  "
                  f"processes used: {seen}")
            rows.append({"job": "gnn", "mode": "dask", "workers": n, "seconds": seconds, "items": len(names),
                         "speedup": serial_s / seconds, "same_answer": same, "worker_processes_seen": seen,
                         **{f"r2_{k}": v for k, v in r2.items()}})

    if "replan" in args.jobs:
        from replan_experiment import run_all

        common = dict(date="2024-06-24", hours=[9, 13, 21], trips=args.trips, seed=0, vehicles=5)
        started = time.perf_counter()
        serial = run_all(**common)
        serial_s = time.perf_counter() - started
        print(f"replan serial: {args.trips} trips in {serial_s:.1f}s")
        rows.append({"job": "replan", "mode": "serial", "workers": 1, "seconds": serial_s, "items": args.trips,
                     "speedup": 1.0, "same_answer": True, "worker_processes_seen": 1})
        for n in ([None] if args.scheduler else args.workers):
            client = cluster.connect(args.scheduler, workers=n or 2)
            try:
                started = time.perf_counter()
                result = run_all(**common, client=client, torch_threads=1)
                seconds = time.perf_counter() - started
            finally:
                cluster.release(client)
            same = compare_replan(serial, result)
            print(f"replan {n or 'cluster'} worker(s): {seconds:.1f}s  speedup x{serial_s / seconds:.2f}  same results: {same}")
            rows.append({"job": "replan", "mode": "dask", "workers": n, "seconds": seconds, "items": args.trips,
                         "speedup": serial_s / seconds, "same_answer": same, "worker_processes_seen": n})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out}  (machine: {os.cpu_count()} logical CPUs)")


if __name__ == "__main__":
    main()
