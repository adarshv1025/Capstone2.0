"""
snapshot_disruptions.py

GNN-level disruptions: return a copy of a traffic snapshot with some edge
features changed, to be scored by the GNN again (GNNCostPredictor.predict).
The model then says how each road responds -- nothing here touches a cost
directly. The world-level alternative (a stated slowdown on chosen edges,
no GNN) is agents/disruptions.py; read its docstring for which to use when.

WHAT THE GNN CAN AND CANNOT MODEL
(measured 2026-10-13, v3 checkpoint, the 2024-06-24 snapshots)
-----------------------------------------------------------------------------
- Rain across the city (with_rain): the model responds in the right
  direction. Network travel time x1.18 at 09:00, x1.21 at 13:00, x1.22 at
  21:00; mean delay probability 0.66 -> 0.88, 0.49 -> 0.67, 0.33 -> 0.49.
  Carbon and EV energy barely move (+1% to +5%). Caveat: rain is rare in the
  training data (weather_impact has mean 0.013 and s.d. 0.073 there, so 0.6
  is about 8 s.d. out) and the test split has no rain snapshot at all, so
  it cannot be checked against labelled rain data. It does match the rule
  that generated the travel-time labels (exact on 97% of segments,
  CLAUDE.md 12.1): x(1 + 0.3 x 0.6) for weather and about +1% for the road
  surface, x1.19 in all.
- A congestion surge on chosen roads (with_congestion_surge): works, but
  only where there is headroom. congestion_index is already 0.98-0.99 on a
  rush-hour snapshot (09:00, 17:00), so a surge changes under 1% there.
  Off-peak it is real but mild: on the surged edges, time x1.14 at 13:00 and
  x1.28 at 21:00, delay probability 0.35 -> 0.46 and 0.26 -> 0.49. Edges
  that were not surged do not change at all.
- The next snapshot of the same day: no helper needed, load it. Note that
  the hour acts as a city-wide multiplier in this dataset (CLAUDE.md 7.2):
  09:00 -> 11:00 is x0.85 on expected time (half of all roads between 0.83
  and 0.89), 15:00 -> 17:00 x1.24 (1.17 to 1.27), 19:00 -> 21:00 x0.62
  (0.61 to 0.71). It re-times a trip; in 80 test trips it never changed
  which road was best by enough to re-route (CLAUDE.md 11.8).
- A road incident via `event_disruption`: DOES NOT WORK. Raising it to 0.6
  or 1.0 on a route's edges made the model predict them 10-24% FASTER and
  less likely to be delayed -- which is what its training labels say:
  segments with event_disruption > 0 are labelled about 25% faster than the
  rule used for the other 97% (CLAUDE.md 12.1). Do not add a helper for it;
  use agents/disruptions.apply_incident instead.
"""

import torch

# Columns of snapshot.edge_attr (CLAUDE.md 2.1).
CONGESTION_INDEX = 12
WEATHER_IMPACT = 13
ROAD_CONDITION_SCORE = 15

RAIN_WEATHER_IMPACT = 0.6
RAIN_ROAD_CONDITION_DROP = 0.06


def with_rain(snapshot, weather_impact: float = RAIN_WEATHER_IMPACT,
              road_condition_drop: float = RAIN_ROAD_CONDITION_DROP):
    """Rain on every road: weather_impact set, road surface condition lowered."""
    out = snapshot.clone()
    out.edge_attr[:, WEATHER_IMPACT] = weather_impact
    out.edge_attr[:, ROAD_CONDITION_SCORE] = (out.edge_attr[:, ROAD_CONDITION_SCORE] - road_condition_drop).clamp(min=0.0)
    out.weather = "Rain"
    return out


def with_congestion_surge(snapshot, edge_rows, level: float = 1.0):
    """
    Congestion raised to `level` on the given edges. edge_rows are row
    numbers of snapshot.edge_attr, which are also the row numbers of the
    costs frame the GNN returns. An edge already above `level` is left alone.
    """
    out = snapshot.clone()
    rows = torch.as_tensor(list(edge_rows), dtype=torch.long)
    out.edge_attr[rows, CONGESTION_INDEX] = out.edge_attr[rows, CONGESTION_INDEX].clamp(min=level)
    return out
