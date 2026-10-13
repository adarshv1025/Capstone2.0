"""
benchmark_sssp.py

Dijkstra vs the Duan et al. "Tsinghua" algorithm (gnn/bmssp.py), measured.

Both are run as single-source searches to EVERY vertex, on the same
adjacency lists, in pure Python, so the comparison is like for like:

  1. the Bengaluru routing graph (needs sample_costs/), four weightings;
  2. synthetic sparse graphs of growing size, to see how the gap moves with
     n: a road-like grid and a random graph with the Bengaluru graph's
     average out-degree (about 2.1).

For each run it records wall time and how many edge relaxations each
algorithm performed, and checks that both return the same distances.

    python gnn/benchmark_sssp.py            # full run, a few minutes
    python gnn/benchmark_sssp.py --quick    # small sizes only

Writes gnn/benchmark_results/sssp_benchmark.csv.
"""

import argparse
import heapq
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn"))

from bmssp import BMSSP  # noqa: E402


def dijkstra(adj, source):
    """Binary-heap Dijkstra with lazy deletion. Returns (dist, relaxations, heap pushes)."""
    dist = {source: 0.0}
    heap = [(0.0, source)]
    done = set()
    relaxations = pushes = 0
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        for v, w in adj.get(u, ()):
            relaxations += 1
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
                pushes += 1
    return dist, relaxations, pushes


def grid_graph(side, rng):
    """Two-way streets on a side x side grid with uneven costs."""
    adj = {}
    for r in range(side):
        for c in range(side):
            u = r * side + c
            adj[u] = []
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < side and 0 <= cc < side:
                    adj[u].append((rr * side + cc, rng.uniform(0.5, 5.0)))
    return adj


def sparse_random_graph(n, rng, avg_out_degree=2.1):
    """A ring (so everything is reachable) plus random edges up to the target degree."""
    adj = {u: [((u + 1) % n, rng.uniform(0.5, 5.0))] for u in range(n)}
    for _ in range(int(n * (avg_out_degree - 1))):
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            adj[u].append((v, rng.uniform(0.5, 5.0)))
    return adj


def bengaluru_graphs():
    costs = sorted((ROOT / "sample_costs").glob("*.csv"))
    if not costs:
        return
    import pandas as pd
    from routing_solver import RoutingSolver
    solver = RoutingSolver(pd.read_csv(costs[0]))
    for name, weights in (
        ("time only", {"time": 1, "delay": 0, "carbon": 0, "ev": 0}),
        ("carbon only", {"time": 0, "delay": 0, "carbon": 1, "ev": 0}),
        ("balanced", {"time": .25, "delay": .25, "carbon": .25, "ev": .25}),
        ("greenest", {"time": .1, "delay": .1, "carbon": .4, "ev": .4}),
    ):
        cost = solver._cost_fn(weights)
        yield f"Bengaluru, {name}", {u: [(v, cost(u, v, d)) for v, d in nbrs.items()] for u, nbrs in solver._G.adj.items()}


def measure(name, adj, sources, repeats):
    n = len(adj)
    m = sum(len(e) for e in adj.values())
    solver = BMSSP(adj, n)
    t_dj, t_bm, relax_dj, relax_bm = [], [], [], []
    stats_sum = {}
    for source in sources:
        best_dj = best_bm = math.inf
        for _ in range(repeats):
            t0 = time.perf_counter()
            ref, r_dj, _ = dijkstra(adj, source)
            best_dj = min(best_dj, time.perf_counter() - t0)
            t0 = time.perf_counter()
            dist, _ = solver.run(source)
            best_bm = min(best_bm, time.perf_counter() - t0)
        if set(dist) != set(ref) or any(not math.isclose(dist[v], d, rel_tol=1e-9, abs_tol=1e-9) for v, d in ref.items()):
            raise AssertionError(f"{name}: BMSSP and Dijkstra disagree from source {source}")
        t_dj.append(best_dj)
        t_bm.append(best_bm)
        relax_dj.append(r_dj)
        relax_bm.append(solver.stats["relaxations"])
        for key, value in solver.stats.items():
            stats_sum[key] = stats_sum.get(key, 0) + value
    log_n = math.log2(n)
    row = {
        "graph": name, "n": n, "m": m, "sources": len(sources),
        "k": solver.k, "t": solver.t, "levels": solver.top_level,
        "dijkstra_ms": statistics.median(t_dj) * 1000,
        "bmssp_ms": statistics.median(t_bm) * 1000,
        "time_ratio": statistics.median(t_bm) / statistics.median(t_dj),
        "dijkstra_relaxations": statistics.mean(relax_dj),
        "bmssp_relaxations": statistics.mean(relax_bm),
        "relaxation_ratio": statistics.mean(relax_bm) / statistics.mean(relax_dj),
        "bmssp_base_cases": stats_sum["base_cases"] / len(sources),
        "bmssp_pulls": stats_sum["pulls"] / len(sources),
        "bmssp_stale_deleted": stats_sum["stale_deleted"] / len(sources),
        # Leading terms of the two bounds, constants ignored: what theory
        # alone says the ratio of work should be at this size.
        "theory_ratio": (m * log_n ** (2 / 3)) / (m + n * log_n),
        "distances_agree": True,
    }
    print(f"{name:34s} n={n:>7,} m={m:>8,} | Dijkstra {row['dijkstra_ms']:9.1f} ms | BMSSP {row['bmssp_ms']:10.1f} ms | "
          f"x{row['time_ratio']:5.1f} slower | relaxations x{row['relaxation_ratio']:4.2f} | theory x{row['theory_ratio']:4.2f}",
          flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small sizes only")
    args = parser.parse_args()
    rng = random.Random(0)
    rows = []

    for name, adj in bengaluru_graphs():
        rows.append(measure(name, adj, random.Random(1).sample(list(adj), 10), repeats=2))

    grid_sides = [32, 64, 128] if args.quick else [32, 64, 128, 256, 512]
    for side in grid_sides:
        adj = grid_graph(side, rng)
        k = 5 if side <= 128 else (3 if side == 256 else 1)
        rows.append(measure(f"grid {side}x{side}", adj, rng.sample(list(adj), k), repeats=1))

    random_sizes = [1_000, 4_000, 16_000] if args.quick else [1_000, 4_000, 16_000, 64_000, 256_000]
    for n in random_sizes:
        adj = sparse_random_graph(n, rng)
        k = 5 if n <= 16_000 else (3 if n == 64_000 else 1)
        rows.append(measure(f"sparse random, degree 2.1", adj, rng.sample(list(adj), k), repeats=1))

    out_dir = ROOT / "gnn" / "benchmark_results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "sssp_benchmark.csv"
    import csv
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out}  ({len(rows)} rows; every row's distances agree with Dijkstra)")


if __name__ == "__main__":
    main()
