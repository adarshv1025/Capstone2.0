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
Weights don't need to sum to 1 -- they're normalized internally. Each raw
cost column is min-max scaled to [0,1] across the full edge set before
combining, since seconds / probability / kg / percent are not comparable
on their raw scales.

USAGE
-----
    solver = RoutingSolver(costs_df)
    route = solver.find_route(
        source=42, target=891,
        weights={"time": 0.5, "delay": 0.2, "carbon": 0.2, "ev": 0.1},
    )
"""

import networkx as nx
import pandas as pd


class RoutingSolver:
    COST_COLUMNS = ["travel_time_s", "delay_probability", "carbon_kg", "ev_energy_pct"]

    def __init__(self, costs_df: pd.DataFrame):
        self.costs_df = costs_df
        # Min-max stats per raw cost column, used to bring the 4 objectives
        # onto a comparable [0,1] scale before combining into one edge weight.
        self._mins = costs_df[self.COST_COLUMNS].min()
        self._maxs = costs_df[self.COST_COLUMNS].max()

    def _normalized(self, col, value):
        lo, hi = self._mins[col], self._maxs[col]
        if hi - lo < 1e-9:
            return 0.0
        return (value - lo) / (hi - lo)

    def build_graph(self, weights: dict) -> nx.DiGraph:
        w = {
            "time": weights.get("time", 0.25),
            "delay": weights.get("delay", 0.25),
            "carbon": weights.get("carbon", 0.25),
            "ev": weights.get("ev", 0.25),
        }
        total_w = sum(w.values()) or 1.0
        w = {k: v / total_w for k, v in w.items()}

        G = nx.DiGraph()
        for row in self.costs_df.itertuples(index=False):
            scalar_cost = (
                w["time"]   * self._normalized("travel_time_s", row.travel_time_s) +
                w["delay"]  * self._normalized("delay_probability", row.delay_probability) +
                w["carbon"] * self._normalized("carbon_kg", row.carbon_kg) +
                w["ev"]     * self._normalized("ev_energy_pct", row.ev_energy_pct)
            )
            G.add_edge(
                row.src, row.dst,
                weight=scalar_cost,
                travel_time_s=row.travel_time_s,
                delay_probability=row.delay_probability,
                carbon_kg=row.carbon_kg,
                ev_energy_pct=row.ev_energy_pct,
            )
        return G

    def find_route(self, source, target, weights: dict = None) -> dict:
        """
        Returns a dict with the path and route-level totals, or
        {"success": False, "reason": ...} if no path exists.

        Note: route_delay_probability is P(at least one edge is delayed) =
        1 - product(1 - p_i) across the route's edges -- NOT a naive sum,
        which isn't statistically valid and can exceed 1.
        """
        weights = weights or {"time": 0.25, "delay": 0.25, "carbon": 0.25, "ev": 0.25}
        G = self.build_graph(weights)

        try:
            path = nx.shortest_path(G, source, target, weight="weight")
        except nx.NetworkXNoPath:
            return {"success": False, "reason": f"No path found from {source} to {target}"}
        except nx.NodeNotFound as e:
            return {"success": False, "reason": str(e)}

        edges = list(zip(path[:-1], path[1:]))
        total_time = sum(G[u][v]["travel_time_s"] for u, v in edges)
        total_carbon = sum(G[u][v]["carbon_kg"] for u, v in edges)
        total_ev = sum(G[u][v]["ev_energy_pct"] for u, v in edges)

        no_delay_prob = 1.0
        for u, v in edges:
            no_delay_prob *= (1.0 - G[u][v]["delay_probability"])
        route_delay_probability = 1.0 - no_delay_prob

        return {
            "success": True,
            "path": path,
            "num_edges": len(edges),
            "total_travel_time_s": total_time,
            "route_delay_probability": route_delay_probability,
            "total_carbon_kg": total_carbon,
            "total_ev_energy_pct": total_ev,
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