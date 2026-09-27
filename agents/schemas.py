"""
schemas.py

Pydantic models shared across the agent layer. These are the contracts every
agent (Route Agent now, Dispatcher/Coordinator/Carbon Optimizer/Execution/
ESG Reporter in later phases) reads and writes, so keep them stable.
"""

from typing import Optional

from pydantic import BaseModel, Field


class DeliveryRequest(BaseModel):
    request_id: str
    origin: int  # graph node id
    destination: int  # graph node id
    priority: str = "balanced"  # fuzzy text: "fastest", "greenest", "balanced", or free-form
    deadline_s: Optional[float] = None  # seconds from dispatch, if any
    package_weight_kg: Optional[float] = None
    vehicle_id: Optional[str] = None  # filled in by the Dispatcher (phase 3+)


class Vehicle(BaseModel):
    vehicle_id: str
    is_ev: bool = False
    battery_capacity_kwh: Optional[float] = None  # required if is_ev
    max_payload_kg: Optional[float] = None
    current_location: Optional[int] = None  # graph node id


class RouteResult(BaseModel):
    request_id: str
    vehicle_id: Optional[str] = None
    success: bool
    reason: Optional[str] = None  # populated when success is False

    path: list[int] = Field(default_factory=list)
    num_edges: int = 0
    total_travel_time_s: float = 0.0
    route_delay_probability: float = 0.0
    total_carbon_kg: float = 0.0
    total_ev_energy_pct: float = 0.0

    priority: str
    weights_used: dict[str, float] = Field(default_factory=dict)
    reasoning: Optional[str] = None  # LLM's rationale for the chosen weights
