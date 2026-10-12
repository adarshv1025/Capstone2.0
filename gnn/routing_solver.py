"""
routing_solver.py

Deterministic, multi-objective routing tool -- the "grounding tool" that
LLM-driven Route Agents call. No LLM, no API key, no provider dependency
here at all; this is plain graph search over the GNN's predicted per-edge
costs. Keeping this step LLM-free means it can be unit tested in isolation
before any agent touches it.

INPUT
-----
costs_df: the DataFrame returned by GNNCostPredictor.predict() (see
gnn_inference.py) -- one row per edge with columns:
    [src, dst, travel_time_s, delay_probability, carbon_kg, ev_energy_pct]

WEIGHTS
-------
A dict of relative importances for the 4 objectives, e.g.:
    {"time": 0.4, "delay": 0.2, "carbon": 0.2, "ev": 0.2}
Weights don't need to sum to 1 -- they're normalized internally. Each
objective is robustly min-max scaled to [0,1] across the edge set before
combining, since seconds / kg / percent are not comparable on their raw
scales.

What each objective measures per edge (changed 2026-10-11, CLAUDE.md 9):
    time   -- EXPECTED travel time: travel_time_s * (1 + DELAY_SLOWDOWN * p),
              i.e. the GNN's time plus what its own delay probability p is
              expected to add.
    delay  -- delay exposure: p * travel_time_s (seconds of travel at risk
              of a delay). Extra weight here means "be more risk-averse than
              the expected time alone".
    carbon -- carbon_kg
    ev     -- ev_energy_pct

USAGE
-----
    solver = RoutingSolver(costs_df)
    route = solver.find_route(
        source=42, target=891,
        weights={"time": 0.5, "delay": 0.2, "carbon": 0.2, "ev": 0.1},
    )
"""

import networkx as nx
import numpy as np
import pandas as pd

# What a delay costs when it happens: that segment takes 30-100% longer.
# Defined once here and imported by the Execution agent's simulation
# (agents/execution_agent.py), so the planner and the simulator make the
# same assumption instead of two constants drifting apart.
DELAY_EXTRA_TIME_RANGE = (1.3, 2.0)
DELAY_SLOWDOWN = sum(DELAY_EXTRA_TIME_RANGE) / 2 - 1.0  # 0.65 = mean extra time fraction


def canonical_edges(costs_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (src, dst): the faster of any parallel edges, original row
    order and index labels preserved.

    The Bengaluru graph has 83 junction pairs joined by two different roads
    (e.g. a 9m link and a 1.2km loop). Every consumer here identifies an
    edge by its (src, dst) pair, so one of the two has to be chosen, and
    chosen the same way everywhere -- use this helper rather than building
    a {(src, dst): row} dict directly. The old behaviour (whichever row came
    last in the file won) kept the SLOWER road in 53 of the 83 pairs, by up
    to 105x.
    """
    if not costs_df.duplicated(["src", "dst"]).any():
        return costs_df
    keep = costs_df.groupby(["src", "dst"], sort=False)["travel_time_s"].idxmin()
    return costs_df.loc[np.sort(keep.to_numpy())]


class RoutingSolver:
    COST_COLUMNS = ["travel_time_s", "delay_probability", "carbon_kg", "ev_energy_pct"]
    OBJECTIVES = ("time", "delay", "carbon", "ev")

    def __init__(self, costs_df: pd.DataFrame, delay_slowdown: float = DELAY_SLOWDOWN):
        self.costs_df = costs_df
        self.delay_slowdown = delay_slowdown

        edges = canonical_edges(costs_df)
        t = edges["travel_time_s"].to_numpy(dtype=float)
        p = edges["delay_probability"].to_numpy(dtype=float)
        carbon = edges["carbon_kg"].to_numpy(dtype=float)
        ev = edges["ev_energy_pct"].to_numpy(dtype=float)
        expected_t = t * (1.0 + delay_slowdown * p)
        # The Execution simulation scales a segment's carbon with its time
        # ratio, so delays inflate carbon by the same expected factor. Used
        # for reporting only -- the "carbon" objective stays the GNN's value.
        expected_carbon = carbon * (1.0 + delay_slowdown * p)
        expected_ev = ev * (1.0 + delay_slowdown * p)  # same reasoning; the Coordinator plans EV range on this

        # Normalized objective values, one column per OBJECTIVES entry.
        #
        # "delay" is p * t, not p alone. Summing a bare probability along a
        # path charges every segment about the same (p sits in a narrow
        # 0.4-0.6 band), so it rewarded routes with FEWER segments rather
        # than less delay: measured on Koramangala -> Hebbal, a high delay
        # weight gave a route 42% slower that was twice as likely to run
        # 30%+ late.
        norm = np.column_stack([
            self._robust_scale(expected_t),
            self._robust_scale(p * t),
            self._robust_scale(carbon),
            self._robust_scale(ev),
        ])

        # Built once per solver. find_route() used to rebuild a weighted
        # DiGraph row by row on every call (~380ms of a ~590ms call, vs ~6ms
        # for the shortest-path search itself); weights are now applied by a
        # cost function over these cached per-edge values instead.
        self._G = nx.DiGraph()
        self._G.add_edges_from(
            (u, v, {
                "norm": n,
                "travel_time_s": ti,
                "expected_time_s": ei,
                "delay_probability": pi,
                "carbon_kg": ci,
                "expected_carbon_kg": xi,
                "ev_energy_pct": vi,
                "expected_ev_energy_pct": wi,
            })
            for u, v, n, ti, ei, pi, ci, xi, vi, wi in zip(
                edges["src"].tolist(), edges["dst"].tolist(), norm.tolist(),
                t.tolist(), expected_t.tolist(), p.tolist(), carbon.tolist(),
                expected_carbon.tolist(), ev.tolist(), expected_ev.tolist(),
            )
        )

    @staticmethod
    def _robust_scale(values: np.ndarray) -> np.ndarray:
        # Robust min-max: the 1st/99th percentiles, not the true min/max.
        # Time, carbon and EV energy are all heavily right-skewed (a handful
        # of outlier edges dwarf the typical edge), so scaling by the true
        # max squeezes the typical edge's normalized value near 0 and lets
        # whichever objective has no long tail dominate any mixed-weight
        # route regardless of its relative weight. Clipping to [p1, p99]
        # before scaling keeps all 4 objectives comparably sensitive to
        # weight changes.
        lo, hi = np.quantile(values, [0.01, 0.99])
        if hi - lo < 1e-9:
            return np.zeros_like(values)
        return (np.clip(values, lo, hi) - lo) / (hi - lo)

    def _cost_fn(self, weights: dict, penalized_edges: set = None, penalty_factor: float = 5.0):
        w = [weights.get(k, 0.25) for k in self.OBJECTIVES]
        total_w = sum(w) or 1.0
        wt, wd, wc, we = (x / total_w for x in w)
        penalized_edges = penalized_edges or set()

        def cost(u, v, d):
            n = d["norm"]
            c = wt * n[0] + wd * n[1] + wc * n[2] + we * n[3]
            return c * penalty_factor if (u, v) in penalized_edges else c

        return cost

    def build_graph(self, weights: dict, penalized_edges: set = None, penalty_factor: float = 5.0) -> nx.DiGraph:
        """
        A copy of the routing graph with each edge's scalar cost for these
        weights stored as its "weight" attribute, for callers that run their
        own graph algorithms (e.g. the Dispatcher's one-to-all distances).
        find_route() does not need this and does not copy the graph.

        penalized_edges: optional set of (src, dst) tuples whose scalar
        cost gets multiplied by penalty_factor -- used by the Coordinator
        (agents/coordinator_agent.py, Phase 4) to steer a re-route away
        from a congested edge without hard-excluding it (a multiplicative
        penalty degrades gracefully if there's no good alternative, instead
        of risking NetworkXNoPath).
        """
        cost = self._cost_fn(weights, penalized_edges, penalty_factor)
        G = self._G.copy()
        for u, v, d in G.edges(data=True):
            d["weight"] = cost(u, v, d)
        return G

    def find_route(
        self, source, target, weights: dict = None, penalized_edges: set = None, penalty_factor: float = 5.0
    ) -> dict:
        """
        Returns a dict with the path and route-level totals, or
        {"success": False, "reason": ...} if no path exists.

        total_travel_time_s is the GNN's delay-free time; expected_travel_time_s
        adds the expected cost of delays and is what a plan should promise.
        expected_carbon_kg and expected_ev_energy_pct are total_carbon_kg and
        total_ev_energy_pct with the same delay allowance.

        Note: route_delay_probability is P(at least one edge is delayed) =
        1 - product(1 - p_i) across the route's edges -- NOT a naive sum,
        which isn't statistically valid and can exceed 1. It is ~1.0 for any
        route of more than a few dozen segments, so prefer
        expected_travel_time_s when comparing routes.
        """
        weights = weights or {"time": 0.25, "delay": 0.25, "carbon": 0.25, "ev": 0.25}
        G = self._G

        try:
            path = nx.shortest_path(G, source, target, weight=self._cost_fn(weights, penalized_edges, penalty_factor))
        except nx.NetworkXNoPath:
            return {"success": False, "reason": f"No path found from {source} to {target}"}
        except nx.NodeNotFound as e:
            return {"success": False, "reason": str(e)}

        edges = list(zip(path[:-1], path[1:]))
        total_time = sum(G[u][v]["travel_time_s"] for u, v in edges)
        expected_time = sum(G[u][v]["expected_time_s"] for u, v in edges)
        total_carbon = sum(G[u][v]["carbon_kg"] for u, v in edges)
        expected_carbon = sum(G[u][v]["expected_carbon_kg"] for u, v in edges)
        total_ev = sum(G[u][v]["ev_energy_pct"] for u, v in edges)
        expected_ev = sum(G[u][v]["expected_ev_energy_pct"] for u, v in edges)

        no_delay_prob = 1.0
        for u, v in edges:
            no_delay_prob *= (1.0 - G[u][v]["delay_probability"])
        route_delay_probability = 1.0 - no_delay_prob

        return {
            "success": True,
            "path": path,
            "num_edges": len(edges),
            "total_travel_time_s": total_time,
            "expected_travel_time_s": expected_time,
            "route_delay_probability": route_delay_probability,
            "total_carbon_kg": total_carbon,
            "expected_carbon_kg": expected_carbon,
            "total_ev_energy_pct": total_ev,
            "expected_ev_energy_pct": expected_ev,
            "weights_used": weights,
        }


# ══════════════════════════════════════════════════════════════════════════
# SMOKE TEST — run against a real snapshot + trained GNN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import glob
    import torch
    from gnn_inference import GNNCostPredictor

    test_files = sorted(glob.glob("./dataset_kaggle/test/snapshot_*.pt"))
    if not test_files:
        raise SystemExit(
            "No test snapshots found at ./dataset_kaggle/test/ -- "
            "update the path above if your data lives elsewhere."
        )

    print(f"Loading a sample snapshot: {test_files[0]}")
    snapshot = torch.load(test_files[0], weights_only=False)

    predictor = GNNCostPredictor()
    costs_df = predictor.predict(snapshot)
    solver = RoutingSolver(costs_df)

    # Pick two nodes that actually appear in the edge list for a real test.
    source = int(costs_df.iloc[0]["src"])
    target = int(costs_df.iloc[-1]["dst"])

    print(f"\nRouting from node {source} to node {target}\n")

    print("Fastest route (time-only):")
    print(solver.find_route(source, target, weights={"time": 1.0, "delay": 0, "carbon": 0, "ev": 0}))

    print("\nGreenest route (carbon-weighted):")
    print(solver.find_route(source, target, weights={"time": 0.1, "delay": 0.1, "carbon": 0.7, "ev": 0.1}))

    print("\nBalanced route (default weights):")
    print(solver.find_route(source, target))
