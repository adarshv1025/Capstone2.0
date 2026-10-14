"""
schemas.py

Pydantic models shared across the agent layer. These are the contracts every
agent (Route Agent now, Dispatcher/Coordinator/Carbon Optimizer/Execution/
ESG Reporter in later phases) reads and writes, so keep them stable.
"""

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class DeliveryRequest(BaseModel):
    request_id: str
    origin: int  # graph node id
    destination: int  # graph node id
    priority: str = "balanced"  # fuzzy text: "fastest", "greenest", "balanced", or free-form
    # Latest acceptable arrival at the destination, in seconds after the
    # vehicle leaves the origin. A hard constraint where given (CLAUDE.md 12).
    deadline_s: Optional[float] = None
    package_weight_kg: Optional[float] = None
    vehicle_id: Optional[str] = None  # filled in by the Dispatcher (phase 3+)


class Vehicle(BaseModel):
    vehicle_id: str
    is_ev: bool = False
    battery_capacity_kwh: Optional[float] = None  # required if is_ev
    max_payload_kg: Optional[float] = None
    current_location: Optional[int] = None  # graph node id
    # A key of agents/vehicles.VEHICLE_TYPES (e.g. "diesel_ldv", "ev_ldv").
    # None = the GNN's own reference vehicle, as before Phase 5. When set,
    # is_ev follows the type and a missing max_payload_kg takes the type's
    # typical payload.
    vehicle_type: Optional[str] = None

    @model_validator(mode="after")
    def _apply_vehicle_type(self):
        if self.vehicle_type is not None:
            from vehicles import profile  # local: vehicles imports nothing from here
            p = profile(self.vehicle_type)
            self.is_ev = p.is_ev
            if self.max_payload_kg is None:
                self.max_payload_kg = p.typical_payload_kg
        return self


class RouteResult(BaseModel):
    request_id: str
    vehicle_id: Optional[str] = None
    success: bool
    reason: Optional[str] = None  # populated when success is False

    path: list[int] = Field(default_factory=list)
    num_edges: int = 0
    total_travel_time_s: float = 0.0  # GNN's delay-free time
    # total_travel_time_s plus the expected cost of delays (RoutingSolver);
    # this is what the plan promises and what Execution measures against.
    expected_travel_time_s: float = 0.0
    route_delay_probability: float = 0.0
    total_carbon_kg: float = 0.0
    expected_carbon_kg: float = 0.0  # total_carbon_kg plus the expected effect of delays
    total_ev_energy_pct: float = 0.0
    # total_ev_energy_pct plus the expected effect of delays -- what the
    # Coordinator checks against the battery.
    expected_ev_energy_pct: float = 0.0

    priority: str
    weights_used: dict[str, float] = Field(default_factory=dict)
    reasoning: Optional[str] = None  # LLM's rationale for the chosen weights

    # Phase 5 (CLAUDE.md 12). With a vehicle_type, every carbon figure above
    # is that vehicle's CO2 (tailpipe, or grid electricity for an EV), not
    # the GNN's reference vehicle.
    vehicle_type: Optional[str] = None
    deadline_s: Optional[float] = None  # copied from the DeliveryRequest
