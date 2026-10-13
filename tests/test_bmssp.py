"""
Tests for gnn/bmssp.py (the Duan et al. "Tsinghua" SSSP algorithm) against a
plain binary-heap Dijkstra. No LLM, no API key, no dataset needed except for
the last test, which is skipped if sample_costs/ is missing.

    pytest tests/test_bmssp.py        or        python tests/test_bmssp.py
"""

import heapq
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn"))

from bmssp import BMSSP, BlockQueue, INF  # noqa: E402


def dijkstra(adj, source):
    dist = {source: 0.0}
    heap = [(0.0, source)]
    done = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        for v, w in adj.get(u, ()):
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def random_graph(n, m, rng, weight):
    adj = {u: [] for u in range(n)}
    for _ in range(m):
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            adj[u].append((v, weight(rng)))
    return adj


def assert_same_distances(adj, source, n=None):
    solver = BMSSP(adj, n)
    dist, pred = solver.run(source)
    ref = dijkstra(adj, source)
    assert set(dist) == set(ref), f"reachable sets differ: {len(dist)} vs {len(ref)}"
    for v, d in ref.items():
        assert math.isclose(dist[v], d, rel_tol=1e-9, abs_tol=1e-9), (v, dist[v], d)
    # pred must describe real paths whose length is the reported distance
    weight = {}
    for u, edges in adj.items():
        for v, w in edges:
            weight[(u, v)] = min(w, weight.get((u, v), math.inf))
    for v in dist:
        if v != source:
            p = pred[v]
            assert math.isclose(dist[v], dist[p] + weight[(p, v)], rel_tol=1e-9, abs_tol=1e-9), v
    assert pred[source] is None
    return solver.stats


WEIGHTS = {
    "floats": lambda r: r.random() * 10,
    "small ints (many ties)": lambda r: float(r.randint(1, 3)),
    "zeros and ones": lambda r: float(r.randint(0, 1)),
    "all equal": lambda r: 1.0,
    "all zero": lambda r: 0.0,
}


def test_tiny_graphs():
    assert_same_distances({0: []}, 0)
    assert_same_distances({0: [(1, 2.0)], 1: []}, 0)
    assert_same_distances({0: [(1, 1.0), (2, 5.0)], 1: [(2, 1.0)], 2: [(0, 1.0)]}, 0)
    assert_same_distances({0: [(1, 0.0)], 1: [(0, 0.0), (2, 0.0)], 2: []}, 0)   # zero-weight cycle
    assert_same_distances({0: [(1, 1.0)], 1: [], 2: [(0, 1.0)]}, 0)               # vertex 2 unreachable


def test_long_chain_with_shortcut_edge():
    # s -> v directly is expensive; the cheap route is a long chain. This is
    # the shape that leaves a stale, larger value for v waiting in a queue.
    n = 400
    adj = {i: [(i + 1, 1.0)] for i in range(n)}
    adj[n] = []
    for target in range(5, n, 7):
        adj[0].append((target, 10.0 * target))
    assert_same_distances(adj, 0)


def test_random_graphs():
    rng = random.Random(7)
    for name, weight in WEIGHTS.items():
        for n, m in [(5, 8), (30, 60), (30, 300), (200, 500), (200, 2000), (1500, 4000), (3000, 6500)]:
            adj = random_graph(n, m, rng, weight)
            for source in rng.sample(range(n), 3):
                assert_same_distances(adj, source, n)


def test_grid_graph():
    side = 45
    rng = random.Random(3)
    adj = {}
    for r in range(side):
        for c in range(side):
            u = r * side + c
            adj[u] = []
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < side and 0 <= cc < side:
                    adj[u].append((rr * side + cc, rng.choice([1.0, 1.0, 2.0, 3.5])))
    for source in (0, side * side // 2, side * side - 1):
        assert_same_distances(adj, source)


def test_block_queue_against_a_sorted_list():
    # Values are (number, 0, key), like the algorithm's (distance, hops,
    # vertex) labels: the key is the tie-breaker, so no two keys ever hold
    # equal values. The queue's contract assumes that.
    rng = random.Random(11)
    for M in (1, 2, 5, 32):
        q = BlockQueue(M, INF)
        model = {}
        floor = 0.0   # everything pulled so far is below this
        for step in range(3000):
            op = rng.random()
            if op < 0.5:
                key = rng.randrange(400)
                value = (floor + rng.random() * 100, 0, key)
                q.insert(key, value)
                if key not in model or value < model[key]:
                    model[key] = value
            elif op < 0.65 and model:
                low = min(model.values())[0]
                if low > floor:
                    pairs = []
                    for _ in range(rng.randint(1, 3 * M)):
                        key = rng.randrange(400, 800)
                        pairs.append((key, (floor + rng.random() * (low - floor) * 0.999, 0, key)))
                    pairs = [(k, v) for k, v in pairs if v < min(model.values())]
                    q.batch_prepend(pairs)
                    for key, value in pairs:
                        if key not in model or value < model[key]:
                            model[key] = value
            elif model:
                x, keys = q.pull()
                expected = sorted(model, key=model.get)[:M]
                assert sorted(keys) == sorted(expected), (M, step)
                pulled_max = max(model[k] for k in keys)
                for k in keys:
                    del model[k]
                assert pulled_max < x
                assert (x == INF) if not model else (x <= min(model.values()))
                floor = pulled_max[0]
            assert len(q) == len(model)


def test_bengaluru_graph():
    costs = sorted((ROOT / "sample_costs").glob("*.csv"))
    if not costs:
        print("  (skipped: no sample_costs/)")
        return
    import pandas as pd
    from routing_solver import RoutingSolver
    solver = RoutingSolver(pd.read_csv(costs[0]))
    rng = random.Random(5)
    nodes = list(solver._G.nodes)
    for weights in ({"time": 1, "delay": 0, "carbon": 0, "ev": 0}, {"time": 0, "delay": 0, "carbon": 1, "ev": 0},
                    {"time": .25, "delay": .25, "carbon": .25, "ev": .25}, {"time": .1, "delay": .1, "carbon": .4, "ev": .4}):
        cost = solver._cost_fn(weights)
        adj = {u: [] for u in nodes}
        for u, v, data in solver._G.edges(data=True):
            adj[u].append((v, cost(u, v, data)))
        for source in rng.sample(nodes, 5):
            assert_same_distances(adj, source, len(nodes))


if __name__ == "__main__":
    import time
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.perf_counter()
            try:
                fn()
                print(f"PASS  {name}  ({time.perf_counter() - t0:.1f}s)")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:300]}")
    raise SystemExit(1 if failed else 0)
