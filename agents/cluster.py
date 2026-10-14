"""
cluster.py

Distributed execution (Phase 5 of the 2026-10-11 plan, CLAUDE.md 12.5) -- the
"over Distributed Computing Systems" in the project title.

Work is sent to worker PROCESSES through a Dask scheduler over TCP. The
workers can be on this machine (connect(workers=N) starts them) or on other
machines (start a scheduler and workers by hand and connect(scheduler=...)).
The same code runs either way; only the address changes.

What is distributed, and why these:
  - score_snapshots(): GNN inference over many traffic snapshots, e.g. the
    whole test split, combined into exact R^2 per target. Data-parallel,
    one snapshot per task.
  - route_batch(): many origin-destination routes on one set of costs.
  - agents/replan_experiment.py --workers N: the closed-loop experiment,
    one trip per task (the heaviest CPU job in the project).
LLM calls are NOT distributed: they wait on the network, not the CPU, and
the pipeline already overlaps them with threads (agents/pipeline.py).

Each worker loads the GNN once and keeps it (worker_predictor()), so the
model is not shipped with every task. Results come back as small summaries,
not 15,083-row frames.

MULTI-MACHINE USE (not tested on two machines -- see CLAUDE.md 12.5)
    machine A:  dask scheduler                         (listens on :8786)
    machine B:  cd <repo>  and set PYTHONPATH=agents;gnn  (":" on Linux/Mac)
                dask worker tcp://<A's address>:8786 --nthreads 1
    anywhere:   python agents/cluster_benchmark.py --scheduler tcp://<A>:8786
Every worker needs its own copy of the repo and of the data it will read
(gnn_checkpoints_v3/, dataset_kaggle/test/), at any path.
"""

import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATHS = [str(ROOT / "agents"), str(ROOT / "gnn")]
for _p in PATHS:
    if _p not in sys.path:
        sys.path.insert(0, _p)

TARGETS = ["travel_time_s", "delay_probability", "carbon_kg", "ev_energy_pct"]


def connect(scheduler: str = None, workers: int = 2, threads_per_worker: int = 1, memory_limit: str = "3GB"):
    """
    A dask.distributed Client. With `scheduler` ("tcp://host:8786"), joins an
    existing cluster. Otherwise starts `workers` local worker processes
    (separate Python processes talking to a local scheduler over TCP).
    """
    from dask.distributed import Client, LocalCluster

    if scheduler:
        return Client(scheduler)
    # Spawned worker processes must be able to import this project's modules.
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(PATHS + ([existing] if existing else []))
    cluster = LocalCluster(n_workers=workers, threads_per_worker=threads_per_worker, processes=True,
                           dashboard_address=None, memory_limit=memory_limit)
    client = Client(cluster)
    client._local_cluster = cluster   # so release() can stop the workers this call started
    return client


def release(client) -> None:
    """
    Disconnects, and stops the worker processes if connect() started them.
    client.close() alone leaves a LocalCluster's workers running (found when
    a 2-worker run's processes were still holding ~1.8 GB as the next run
    started 4 more); a cluster joined by address is left running.
    """
    local = getattr(client, "_local_cluster", None)
    client.close()
    if local is not None:
        local.close()


def worker_name() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


# ── worker-side state: one copy per worker process ───────────────────────
_STATE: dict = {}


def worker_predictor(torch_threads: int = None):
    """The GNN, loaded once per worker process and kept."""
    if "predictor" not in _STATE:
        import contextlib
        import io

        import torch

        if torch_threads:
            torch.set_num_threads(torch_threads)
        from gnn_inference import GNNCostPredictor

        with contextlib.redirect_stdout(io.StringIO()):   # its load messages, once per worker
            _STATE["predictor"] = GNNCostPredictor(checkpoint_dir=str(ROOT / "gnn_checkpoints_v3"), device="cpu")
    return _STATE["predictor"]


def score_snapshot(snapshot_name: str, torch_threads: int = None) -> dict:
    """
    GNN inference on one test snapshot (a file name in dataset_kaggle/test,
    resolved on the worker's own copy). Returns sufficient statistics for an
    exact pooled R^2 per target, plus timing and which worker ran it.
    """
    import numpy as np
    import torch

    started = time.perf_counter()
    snapshot = torch.load(ROOT / "dataset_kaggle" / "test" / snapshot_name, weights_only=False)
    costs = worker_predictor(torch_threads).predict(snapshot)
    pred = costs[TARGETS].to_numpy(dtype=np.float64)
    y = snapshot.y.numpy().astype(np.float64)
    return {
        "snapshot": snapshot_name, "n": int(len(y)),
        "sum_y": y.sum(axis=0).tolist(), "sum_y2": (y ** 2).sum(axis=0).tolist(),
        "sse": ((y - pred) ** 2).sum(axis=0).tolist(),
        "seconds": time.perf_counter() - started, "worker": worker_name(),
    }


def combine_r2(parts: list[dict]) -> dict:
    """Pooled R^2 per target from score_snapshot() results (any order, any split)."""
    n = sum(p["n"] for p in parts)
    out = {}
    for k, name in enumerate(TARGETS):
        sum_y = sum(p["sum_y"][k] for p in parts)
        sum_y2 = sum(p["sum_y2"][k] for p in parts)
        sse = sum(p["sse"][k] for p in parts)
        sst = sum_y2 - sum_y ** 2 / n
        out[name] = 1.0 - sse / sst
    return out


def score_snapshots(client, snapshot_names: list[str], torch_threads: int = None) -> list[dict]:
    """Distributes score_snapshot over the cluster; results in input order."""
    futures = client.map(score_snapshot, snapshot_names, torch_threads=torch_threads, pure=False)
    return client.gather(futures)


def route_one(costs_csv: str, source: int, target: int, weights: dict) -> dict:
    """One route on a costs CSV (cached per worker). Needs no torch."""
    import pandas as pd

    from routing_solver import RoutingSolver

    key = ("solver", costs_csv)
    if key not in _STATE:
        _STATE[key] = RoutingSolver(pd.read_csv(costs_csv))
    r = _STATE[key].find_route(source, target, weights=weights)
    return {"source": source, "target": target, "success": r["success"], "path": r.get("path", []),
            "expected_travel_time_s": r.get("expected_travel_time_s"), "worker": worker_name()}


def route_batch(client, costs_csv: str, pairs: list[tuple], weights: dict) -> list[dict]:
    """Routes every (source, target) pair across the cluster; results in input order."""
    futures = [client.submit(route_one, costs_csv, s, t, weights, pure=False) for s, t in pairs]
    return client.gather(futures)
