"""
dispatcher_agent.py

Phase 3: Dispatcher/Planner (CLAUDE.md roster #1). Assigns a batch of
DeliveryRequests to available Vehicles before routing starts.

Kept deterministic, no LLM: vehicle assignment here is a nearest-capable-
vehicle matching problem with a clear objective, not a fuzzy judgment call.
The LLM budget is spent where judgment is actually needed -- Route Agent's
priority interpretation (Phase 2) and the Coordinator/Carbon Optimizer's
negotiation (Phase 4-5). See CLAUDE.md 3.2.

Added in Phase 5 of the 2026-10-11 plan (CLAUDE.md 12):
- Carbon-aware choice of vehicle. When any vehicle has a vehicle_type, a
  cleaner vehicle can win a delivery even if it is a little further away:
  each (vehicle, delivery) pair is scored as
      minutes to reach the pickup + CARBON_PRICE_S_PER_KG x trip CO2
  where trip CO2 is that vehicle's CO2 on the fastest route from pickup to
  drop-off. With no vehicle types the score is the distance alone, exactly
  as before.
- plan_tours(): several deliveries per vehicle, ordered by OR-Tools
  (agents/multi_stop.py).
"""

import sys
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gnn"))
from routing_solver import RoutingSolver  # noqa: E402

from schemas import DeliveryRequest, Vehicle
from vehicles import vehicle_co2_kg

# How much extra time to reach a pickup is worth one kg of CO2 saved on the
# delivery: a vehicle up to 10 minutes further away wins if it emits 1 kg
# less. A policy setting, not a measured value.
CARBON_PRICE_S_PER_KG = 600.0


class DispatcherAgent:
    """
    Assigns each DeliveryRequest to the nearest capable, available Vehicle,
    using time-only shortest-path distance from the vehicle's current
    location to the request's pickup (origin), and respecting payload
    capacity.

    Assignment is greedy: every feasible (vehicle, request) pair is sorted
    by distance ascending and assigned first-come-first-served, so no
    vehicle or request is double-booked. This is not a globally optimal
    assignment (that would need something like the Hungarian algorithm),
    but it's deterministic, explainable, and enough for fleet sizes at this
    project's scale.
    """

    def __init__(self, costs_df: pd.DataFrame, carbon_price_s_per_kg: float = CARBON_PRICE_S_PER_KG):
        self.costs_df = costs_df
        self.solver = RoutingSolver(costs_df)
        # Time-only graph: dispatch cares about "how long to reach the
        # pickup", not the multi-objective cost Route Agents optimize for.
        self.graph = self.solver.build_graph({"time": 1.0, "delay": 0.0, "carbon": 0.0, "ev": 0.0})
        self.carbon_price_s_per_kg = carbon_price_s_per_kg
        # One entry per assignment made by the last assign() call: who got
        # what and why (time to pickup, trip CO2, score). For display.
        self.last_assignments: list[dict] = []

    def assign(
        self, requests: list[DeliveryRequest], vehicles: list[Vehicle]
    ) -> tuple[list[DeliveryRequest], list[DeliveryRequest]]:
        """
        Returns (assigned, unassigned). `assigned` requests have
        `vehicle_id` populated in place; `unassigned` had no feasible
        vehicle (no path to pickup, or no vehicle with enough capacity).
        """
        # One Dijkstra run per vehicle location beats one per
        # (vehicle, request) pair.
        distances_by_vehicle: dict[str, dict] = {}
        for vehicle in vehicles:
            if vehicle.current_location is None or vehicle.current_location not in self.graph:
                distances_by_vehicle[vehicle.vehicle_id] = {}
                continue
            distances_by_vehicle[vehicle.vehicle_id] = nx.single_source_dijkstra_path_length(
                self.graph, vehicle.current_location, weight="weight"
            )

        # Carbon-aware only when the fleet has vehicle types (see module docstring).
        typed = any(v.vehicle_type for v in vehicles)
        trip_reference = {}  # request_id -> (carbon_kg, ev_energy_pct) of the fastest pickup -> drop-off route
        if typed:
            for request in requests:
                fastest = self.solver.find_route(request.origin, request.destination,
                                                 weights={"time": 1.0, "delay": 0.0, "carbon": 0.0, "ev": 0.0})
                if fastest["success"]:
                    trip_reference[request.request_id] = (fastest["expected_carbon_kg"], fastest["expected_ev_energy_pct"])

        candidates = []  # (score, request, vehicle, time_to_pickup_s, trip_co2_kg)
        for request in requests:
            for vehicle in vehicles:
                if (
                    vehicle.max_payload_kg is not None
                    and request.package_weight_kg is not None
                    and request.package_weight_kg > vehicle.max_payload_kg
                ):
                    continue
                dist = distances_by_vehicle.get(vehicle.vehicle_id, {}).get(request.origin)
                if dist is None:
                    continue
                trip_co2 = None
                score = dist
                if typed and request.request_id in trip_reference:
                    trip_co2 = vehicle_co2_kg(vehicle.vehicle_type, *trip_reference[request.request_id])
                    score = dist + self.carbon_price_s_per_kg * trip_co2
                candidates.append((score, request, vehicle, dist, trip_co2))

        candidates.sort(key=lambda c: c[0])

        assigned_vehicles: set[str] = set()
        assigned_requests: set[str] = set()
        assigned: list[DeliveryRequest] = []

        self.last_assignments = []
        for score, request, vehicle, dist, trip_co2 in candidates:
            if request.request_id in assigned_requests or vehicle.vehicle_id in assigned_vehicles:
                continue
            request.vehicle_id = vehicle.vehicle_id
            assigned_requests.add(request.request_id)
            assigned_vehicles.add(vehicle.vehicle_id)
            assigned.append(request)
            self.last_assignments.append({
                "request_id": request.request_id, "vehicle_id": vehicle.vehicle_id,
                "vehicle_type": vehicle.vehicle_type, "time_to_pickup_s": dist,
                "trip_co2_kg": trip_co2, "score": score,
            })

        unassigned = [r for r in requests if r.request_id not in assigned_requests]
        return assigned, unassigned

    def plan_tours(self, depot: int, requests: list[DeliveryRequest], vehicles: list[Vehicle],
                   weights: dict = None, **kwargs) -> dict:
        """
        Several deliveries per vehicle: which vehicle takes which stops, and
        in what order, solved with OR-Tools. All vehicles start at `depot`;
        each request's destination is a stop. See agents/multi_stop.py.
        """
        from multi_stop import plan_tours  # imports OR-Tools only when used

        return plan_tours(self.costs_df, depot, requests, vehicles, weights=weights, **kwargs)


# ══════════════════════════════════════════════════════════════════════════
# SMOKE TEST -- fully deterministic, runs against sample_costs/*.csv,
# no LLM/API key needed.
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import glob

    sample_costs_dir = Path(__file__).resolve().parent.parent / "sample_costs"
    csv_files = sorted(glob.glob(str(sample_costs_dir / "*.csv")))
    if not csv_files:
        raise SystemExit(f"No sample cost CSVs found in {sample_costs_dir}")

    costs_df = pd.read_csv(csv_files[0])
    print(f"Loaded {csv_files[0]} ({len(costs_df)} edges)")

    nodes = pd.unique(costs_df[["src", "dst"]].values.ravel())

    requests = [
        DeliveryRequest(request_id=f"req-{i}", origin=int(nodes[i * 7]), destination=int(nodes[-(i + 1)]), package_weight_kg=50.0)
        for i in range(4)
    ]
    vehicles = [
        Vehicle(vehicle_id=f"veh-{i}", max_payload_kg=200.0, current_location=int(nodes[i * 3]))
        for i in range(3)  # fewer vehicles than requests, so one goes unassigned
    ]

    dispatcher = DispatcherAgent(costs_df)
    assigned, unassigned = dispatcher.assign(requests, vehicles)

    print(f"\n{len(assigned)} assigned, {len(unassigned)} unassigned\n")
    for r in assigned:
        print(f"  {r.request_id} -> {r.vehicle_id}")
    for r in unassigned:
        print(f"  {r.request_id} -> UNASSIGNED")
