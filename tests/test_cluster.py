"""
Tests for the distributed runtime (agents/cluster.py, CLAUDE.md 12.5).

The main test starts a Dask scheduler and two workers as SEPARATE operating
system processes from the command line -- exactly what would run on separate
machines -- connects to them over TCP, sends routing work, and checks that
the answers equal a serial run and that both workers did some of it.

No LLM and no torch (the work is routing on sample_costs/). Takes about
20-30 seconds, most of it starting processes.

    python tests/test_cluster.py        (or: pytest tests/)
"""

import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agents"))
sys.path.insert(0, str(ROOT / "gnn"))

import cluster  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def sample_pairs(csv: str, n: int = 24):
    import pandas as pd

    costs = pd.read_csv(csv)
    nodes = sorted(set(costs["src"]) | set(costs["dst"]))
    rng = random.Random(0)
    return [tuple(rng.sample(nodes, 2)) for _ in range(n)]


def test_combine_r2_is_exact_for_any_split():
    import numpy as np

    rng = np.random.default_rng(0)
    y = rng.normal(size=(1000, 4))
    pred = y + rng.normal(scale=0.3, size=y.shape)

    def part(a, b):
        yy, pp = y[a:b], pred[a:b]
        return {"n": len(yy), "sum_y": yy.sum(0).tolist(), "sum_y2": (yy ** 2).sum(0).tolist(),
                "sse": ((yy - pp) ** 2).sum(0).tolist()}

    whole = cluster.combine_r2([part(0, 1000)])
    split = cluster.combine_r2([part(0, 137), part(137, 600), part(600, 1000)])
    direct = 1 - ((y - pred) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)
    for k, name in enumerate(cluster.TARGETS):
        assert abs(whole[name] - split[name]) < 1e-12
        assert abs(whole[name] - direct[k]) < 1e-12


def test_separate_processes_over_tcp():
    from dask.distributed import Client

    csv = str(sorted((ROOT / "sample_costs").glob("*.csv"))[0])
    port = free_port()
    address = f"tcp://127.0.0.1:{port}"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(cluster.PATHS)}
    dask_cli = [sys.executable, "-m", "distributed.cli.dask_scheduler"]
    worker_cli = [sys.executable, "-m", "distributed.cli.dask_worker"]
    procs = [subprocess.Popen(dask_cli + ["--host", "127.0.0.1", "--port", str(port), "--no-dashboard"],
                              env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    try:
        time.sleep(3)
        for _ in range(2):
            procs.append(subprocess.Popen(worker_cli + [address, "--nthreads", "1", "--no-dashboard"],
                                          env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        client = Client(address, timeout=30)
        try:
            client.wait_for_workers(2, timeout=60)
            pairs = sample_pairs(csv)
            weights = {"time": 0.25, "delay": 0.25, "carbon": 0.25, "ev": 0.25}
            distributed = cluster.route_batch(client, csv, pairs, weights)
        finally:
            client.close()
    finally:
        for p in procs[::-1]:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()

    serial = [cluster.route_one(csv, s, t, weights) for s, t in pairs]
    assert [r["path"] for r in distributed] == [r["path"] for r in serial]
    workers = {r["worker"] for r in distributed}
    assert len(workers) == 2, workers                     # both worker processes did some of it
    assert str(os.getpid()) not in {w.split(":")[1] for w in workers}   # and not this process


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        started = time.perf_counter()
        try:
            fn()
            print(f"PASS  {name}  ({time.perf_counter() - started:.1f}s)")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}  {e!r}")
    print(f"\n{len(tests) - failed} of {len(tests)} passed")
    sys.exit(1 if failed else 0)
