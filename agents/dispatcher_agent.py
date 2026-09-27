"""
dispatcher_agent.py

Phase 3: Dispatcher/Planner (CLAUDE.md roster #1). Assigns a batch of
DeliveryRequests to available Vehicles before routing starts.

Kept deterministic, no LLM: vehicle assignment here is a nearest-capable-
vehicle matching problem with a clear objective, not a fuzzy judgment call.
The LLM budget is spent where judgment is actually needed -- Route Agent's
priority interpretation (Phase 2) and the Coordinator/Carbon Optimizer's
negotiation (Phase 4-5). See CLAUDE.md 3.2.
"""

import sys
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gnn"))
from routing_solver import RoutingSolver  # noqa: E402

from schemas import DeliveryRequest, Vehicle


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

    def __init__(self, costs_df: pd.DataFrame):
        solver = RoutingSolver(costs_df)
        # Time-only graph: dispatch cares about "how long to reach the
        # pickup", not the multi-objective cost Route Agents optimize for.
        self.graph = solver.build_graph({"time": 1.0, "delay": 0.0, "carbon": 0.0, "ev": 0.0})

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

        candidates = []  # (distance, request, vehicle)
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
                candidates.append((dist, request, vehicle))

        candidates.sort(key=lambda c: c[0])

        assigned_vehicles: set[str] = set()
        assigned_requests: set[str] = set()
        assigned: list[DeliveryRequest] = []

        for _, request, vehicle in candidates:
            if request.request_id in assigned_requests or vehicle.vehicle_id in assigned_vehicles:
                continue
            request.vehicle_id = vehicle.vehicle_id
            assigned_requests.add(request.request_id)
            assigned_vehicles.add(vehicle.vehicle_id)
            assigned.append(request)

        unassigned = [r for r in requests if r.request_id not in assigned_requests]
        return assigned, unassigned


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
