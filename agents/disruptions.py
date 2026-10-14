"""
disruptions.py

Things that change on the road AFTER a vehicle has left, so the closed loop
(agents/feedback_loop.py) has something to react to. There is no live
traffic feed in this project (CLAUDE.md 8.1), so a disruption is modelled as
a second set of per-edge costs: the world the plan was made on, and the
world after the change.

Two kinds, and it matters which one a result came from:

1. GNN-level -- edit the snapshot's features and run the GNN again. The
   model itself says how the roads respond. Lives in
   gnn/snapshot_disruptions.py (needs torch): rain across the city, a
   congestion surge on chosen roads, or simply the next snapshot of the day.

2. World-level -- this file (no torch). A road incident: the affected edges
   get slower by a stated factor. This is an ASSUMPTION about the world, not
   a GNN prediction. It exists because the GNN cannot model an incident:
   measured 2026-10-13, raising `event_disruption` on a route's edges made
   the model predict those edges 10-24% FASTER and less likely to be delayed.
   That follows the data: in the training labels, segments with
   event_disruption > 0 take about 25% LESS time than the rule that sets the
   other 97% of segments (CLAUDE.md 12.1). Say "we assume a 4x slowdown"
   when showing it.
"""

from collections import defaultdict, deque

import pandas as pd

# A blocked or half-blocked junction: traffic still gets through, slowly.
INCIDENT_SLOWDOWN = 4.0
INCIDENT_DELAY_PROBABILITY = 0.9


def edge_at_fraction(path: list, edge_time: dict, fraction: float) -> tuple:
    """
    The edge of `path` that a vehicle is on after `fraction` (0-1) of the
    path's travel time. edge_time maps (src, dst) -> seconds.
    """
    edges = list(zip(path[:-1], path[1:]))
    if not edges:
        raise ValueError("path has no edges")
    target = fraction * sum(edge_time[e] for e in edges)
    elapsed = 0.0
    for e in edges:
        elapsed += edge_time[e]
        if elapsed >= target:
            return e
    return edges[-1]


def edges_near(costs_df: pd.DataFrame, center_edge: tuple, hops: int = 1) -> set:
    """
    The road segments an incident on `center_edge` affects: every edge, in
    both directions, between two junctions that are within `hops` junctions
    of either end of center_edge. hops=0 is the segment itself (both
    directions); hops=1 adds the junctions next to it, and so on.
    """
    neighbours = defaultdict(set)
    pairs = set(zip(costs_df["src"].tolist(), costs_df["dst"].tolist()))
    for u, v in pairs:
        neighbours[u].add(v)
        neighbours[v].add(u)

    depth = {center_edge[0]: 0, center_edge[1]: 0}
    queue = deque(depth)
    while queue:
        node = queue.popleft()
        if depth[node] == hops:
            continue
        for other in neighbours[node]:
            if other not in depth:
                depth[other] = depth[node] + 1
                queue.append(other)
    return {(u, v) for u, v in pairs if u in depth and v in depth}


def apply_incident(
    costs_df: pd.DataFrame, edges: set,
    slowdown: float = INCIDENT_SLOWDOWN, delay_probability: float = INCIDENT_DELAY_PROBABILITY,
) -> pd.DataFrame:
    """
    A copy of costs_df with an incident on `edges` (a set of (src, dst)).
    Travel time is multiplied by `slowdown`; carbon and EV energy by the same
    factor, the same rule the Execution agent's noise model uses (time spent
    crawling burns fuel and battery in proportion); delay probability is
    raised to at least `delay_probability`.
    """
    out = costs_df.copy()
    hit = pd.Series(list(zip(out["src"].tolist(), out["dst"].tolist())), index=out.index).isin(edges)
    out.loc[hit, ["travel_time_s", "carbon_kg", "ev_energy_pct"]] *= slowdown
    out.loc[hit, "delay_probability"] = out.loc[hit, "delay_probability"].clip(lower=delay_probability)
    return out
