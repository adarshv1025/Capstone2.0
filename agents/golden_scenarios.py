"""
golden_scenarios.py

Phase 8: the fixed golden scenario set the evaluation harness
(evaluation_harness.py) scores every system against (CLAUDE.md 3.12).
Chosen from cases already found interesting during Phases 3-6 testing, not
just happy paths, and derived deterministically from a costs_df rather than
hand-typed node IDs -- but still "fixed" in the meaningful sense, because
graph topology is stable across every sample_costs/*.csv snapshot (verified:
all 5 have identical (src, dst) columns in identical order; only the
traffic-dependent predicted values differ per snapshot). So the same
scenario builder produces the same origin/destination pair no matter which
snapshot is passed in, and the same scenario can be run against multiple
snapshots to see how each system holds up under different traffic
conditions.

Each builder returns (requests, vehicles) with fresh model instances on
every call -- callers must call the builder again for each system they run
a scenario against, not share/reuse one pair of lists across systems, since
DispatcherAgent.assign() mutates request.vehicle_id in place.
"""

import pandas as pd

from schemas import DeliveryRequest, Vehicle


def edge_overload_scenario(costs_df: pd.DataFrame) -> tuple[list[DeliveryRequest], list[Vehicle]]:
    """
    3 requests sharing one origin/destination -- guarantees a shared-edge
    conflict for the Coordinator to resolve. This exact pair (first row's
    src -> last row's dst) is the one used in coordinator_agent.py's own
    smoke test and in CLAUDE.md 3.8's bottleneck-edge trace, where one edge
    stayed overloaded no matter how many negotiation rounds ran -- a
    realistic stress case, not just "conflicts exist."
    """
    origin = int(costs_df.iloc[0]["src"])
    destination = int(costs_df.iloc[-1]["dst"])
    requests = [
        DeliveryRequest(request_id=f"eo-{i}", origin=origin, destination=destination, priority="balanced")
        for i in range(3)
    ]
    vehicles = [Vehicle(vehicle_id=f"eo-veh-{i}", current_location=origin) for i in range(3)]
    return requests, vehicles


def ev_infeasible_scenario(costs_df: pd.DataFrame) -> tuple[list[DeliveryRequest], list[Vehicle]]:
    """
    One EV request with a battery far too small for the trip (1.0kWh --
    every route observed on this pair across Phase 4/5/6 testing used
    several kWh even under the most energy-conscious weighting tried).
    Expected to stay a hard EV violation no matter how the Coordinator's
    reweighting or the Execution simulation goes -- a genuine infeasibility
    case (CLAUDE.md 3.8's ev_energy_pct-correlates-with-distance finding),
    not a tuning failure the harness should expect any system to "solve."
    """
    nodes = pd.unique(costs_df[["src", "dst"]].values.ravel())
    origin, destination = int(nodes[100]), int(nodes[6000])
    request = DeliveryRequest(request_id="ev-0", origin=origin, destination=destination, priority="fastest")
    vehicle = Vehicle(vehicle_id="ev-veh-0", is_ev=True, battery_capacity_kwh=1.0, current_location=origin)
    return [request], [vehicle]


def ev_feasible_scenario(costs_df: pd.DataFrame) -> tuple[list[DeliveryRequest], list[Vehicle]]:
    """
    One EV request whose battery is tight but genuinely feasible IF the
    system routes energy-consciously -- unlike ev_infeasible_scenario
    (which only exercises the hard-rejection path), this exercises whether
    EV-aware routing actually helps.

    This origin/destination pair was chosen (not the ev_infeasible one)
    because it has real EV route diversity: a pure time-weighted route
    needs ~2.83kWh while the most energy-conscious route found needs only
    ~2.34kWh (a 17% spread, with different edge counts -- 54 vs 44 -- so
    it's a genuinely different path, not just noise). Several other pairs
    tried during scenario design had under 1% spread -- CLAUDE.md 3.8's
    finding that ev_energy_pct correlates strongly with route distance
    means most pairs don't have much EV-specific route diversity to
    exploit, so this pair had to be found by scanning several candidates,
    not assumed.

    IMPORTANT CALIBRATION NOTE, found while testing this scenario: hard EV
    feasibility (ExecutionAgent.hard_ev_violation) is checked against
    SIMULATED ACTUAL energy use, not the planned figures above -- and
    because carbon/EV energy scale with the realized time ratio per edge
    (3.10's noise model), a route with several delayed edges can use
    30-50%+ more energy than planned. A battery set between the two PLANNED
    figures (e.g. 2.5kWh) was tested and found ev_feasibility_rate=0.0 for
    EVERY system, including the EV-aware ones -- the noise inflation
    swallowed the entire 17% planned-route margin. Sampling 5 seeds' actual
    simulated kWh for both extremes showed why a single threshold can't
    perfectly separate them every run: worst-case actual ranged ~3.52-4.08
    kWh, best-case actual ranged ~2.71-3.46 kWh, with real overlap between
    them (e.g. one seed: worst=3.517 vs best=3.374, barely 4% apart).
    battery_capacity_kwh=3.3 sits between the two ranges' typical centers
    (~3.72 vs ~3.16) for the best *typical* differentiation, but -- unlike
    edge_overload/ev_infeasible, which are deterministic regardless of
    noise -- this scenario's outcome is genuinely seed-sensitive: EV-aware
    routing is more LIKELY to succeed here than naive routing, not
    guaranteed on every run. That's an honest reflection of real EV range
    anxiety (delay exposure, not just route choice, decides feasibility),
    not a scenario-design flaw to eliminate.
    """
    nodes = pd.unique(costs_df[["src", "dst"]].values.ravel())
    origin, destination = int(nodes[300]), int(nodes[6500])
    request = DeliveryRequest(request_id="evf-0", origin=origin, destination=destination, priority="greenest")
    vehicle = Vehicle(vehicle_id="evf-veh-0", is_ev=True, battery_capacity_kwh=3.3, current_location=origin)
    return [request], [vehicle]


def large_batch_scenario(costs_df: pd.DataFrame) -> tuple[list[DeliveryRequest], list[Vehicle]]:
    """
    12 requests across 12 verified-connected origin/destination pairs
    (CLAUDE.md 4.2 item 3: stress-test the Dispatcher/Coordinator at a more
    realistic fleet scale than the 3-5 requests every other scenario here
    uses). Rotating priorities and a mixed EV/non-EV fleet, same style as
    mixed_priority_scenario but roughly 2.5x the batch size.
    """
    nodes = pd.unique(costs_df[["src", "dst"]].values.ravel())
    # (origin_idx, destination_idx) pairs -- verified connected at scenario
    # design time; index 1650 (a natural fit for the spread pattern below)
    # turned out to have no outgoing path to that side of the graph, so
    # 1700 is used in its place.
    pairs = [
        (0, 6900), (550, 6350), (1100, 5800), (1700, 5250), (2200, 4700),
        (2750, 4150), (3300, 3600), (3850, 3050), (4400, 2500), (4950, 1950),
        (5500, 1400), (6050, 850),
    ]
    priorities = ["fastest", "greenest", "balanced"]
    requests = [
        DeliveryRequest(
            request_id=f"lb-{i}",
            origin=int(nodes[a]),
            destination=int(nodes[b]),
            priority=priorities[i % len(priorities)],
            package_weight_kg=50.0,
        )
        for i, (a, b) in enumerate(pairs)
    ]
    vehicles = [
        Vehicle(
            vehicle_id=f"lb-veh-{i}",
            is_ev=(i % 3 == 0),
            battery_capacity_kwh=50.0,
            max_payload_kg=200.0,
            current_location=int(nodes[pairs[i][0]]),
        )
        for i in range(len(pairs))
    ]
    return requests, vehicles


def mixed_priority_scenario(costs_df: pd.DataFrame) -> tuple[list[DeliveryRequest], list[Vehicle]]:
    """
    5 requests across 5 different origin/destination pairs with rotating
    priorities -- the "typical" batch used throughout Phase 3-7 smoke tests
    (pipeline.py, esg_reporter_agent.py), now formalized as a golden
    scenario so those ad hoc test batches become part of the same fixed
    comparison set instead of one-off throwaway data.
    """
    nodes = pd.unique(costs_df[["src", "dst"]].values.ravel())
    priorities = ["fastest", "greenest", "balanced", "fastest", "balanced"]
    requests = [
        DeliveryRequest(
            request_id=f"mp-{i}",
            origin=int(nodes[i * 7]),
            destination=int(nodes[-(i + 1)]),
            priority=priorities[i],
            package_weight_kg=50.0,
        )
        for i in range(5)
    ]
    vehicles = [
        Vehicle(
            vehicle_id=f"mp-veh-{i}",
            is_ev=(i % 2 == 0),
            battery_capacity_kwh=50.0,
            max_payload_kg=200.0,
            current_location=int(nodes[i * 3]),
        )
        for i in range(5)
    ]
    return requests, vehicles


# name -> builder(costs_df) -> (requests, vehicles). The evaluation harness
# iterates this in order.
GOLDEN_SCENARIOS = [
    ("edge_overload", edge_overload_scenario),
    ("ev_infeasible", ev_infeasible_scenario),
    ("ev_feasible", ev_feasible_scenario),
    ("mixed_priority", mixed_priority_scenario),
    ("large_batch", large_batch_scenario),
]
