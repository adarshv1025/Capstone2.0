"""
vehicles.py

Vehicle-specific carbon (Phase 5 of the 2026-10-11 plan, CLAUDE.md 12).

The GNN predicts ONE carbon figure per road segment, for one reference
vehicle. Checked on the test snapshots (CLAUDE.md 12.1): base_carbon is
0.27 kg CO2 per km of road on every segment (1st-99th percentile 0.267 to
0.273), the carbon label is exactly
    base_carbon x (1 + 0.4 congestion) x (1 + 0.05 weather)
on about 95% of segments, and the EV label is exactly
    0.25 kWh per km x (1 + 0.5 congestion)   (as % of a 50 kWh battery)
on all of them. So the reference vehicle emits 0.27 kg CO2/km in
free-flowing traffic and the reference EV uses 0.25 kWh/km.

A real fleet is not one vehicle. This module scales the GNN's figures to a
vehicle type:
  - combustion vehicles: carbon x (type's kg CO2/km / 0.27). The congestion
    and weather effects the GNN learned are kept; only the level changes.
  - electric vehicles: no tailpipe CO2. Their CO2 is the electricity they
    use x the grid's emission factor (CEA, see below).

SOURCES
-------
Combustion factors (kg CO2 per km), India GHG Program, "India Specific Road
Transport Emission Factors", Version 1.0, 2015 (WRI India, CII, TERI):
https://indiaghgp.org/sites/default/files/Road%20Transport%20Technical%20Paper.pdf
    Freight vehicles: LDV (<3.5 t) 0.3070, MDV (<12 t) 0.5928, HDV (>12 t) 0.7375
    Three-wheelers: petrol 0.1135, diesel 0.1322, CNG 0.10768
    Two-wheelers: scooter <110 cc 0.0334
They are derived from rated fuel efficiency (km per litre), so they are
compared with the GNN's free-flow 0.27, not with its congested average.

Grid emission factor: Central Electricity Authority, "CO2 Baseline Database
for the Indian Power Sector", User Guide Version 20.0 (December 2024),
Table S: weighted average for FY 2023-24 = 0.727 t CO2/MWh = 0.727 kg/kWh
(adjusted for cross-border transfers, including renewables and captive
injection).

WHAT THIS IS NOT
----------------
- Travel time is the same for every vehicle type: the GNN has one time per
  road. A scooter is not modelled as faster in traffic, nor a truck as
  slower.
- The two kinds of figure are not like for like: combustion is tailpipe
  only (tank-to-wheel), electric is electricity at the power station, with
  no transmission or charging losses. Both are stated as such wherever they
  are shown.
- The electric light goods vehicle uses the dataset's own 0.25 kWh/km. That
  is the GNN's assumption, not a measured figure for a particular model.
- Payloads are typical values used as defaults when a Vehicle has no
  max_payload_kg; they are not from the sources above.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gnn"))

# The GNN's reference vehicle (CLAUDE.md 12.1).
REFERENCE_KG_CO2_PER_KM = 0.27
REFERENCE_KWH_PER_KM = 0.25
EV_REFERENCE_KWH = 50.0             # ev_energy_pct is % of a 50 kWh battery (CLAUDE.md 2.1)
GRID_KG_CO2_PER_KWH = 0.727         # CEA v20.0, FY 2023-24 weighted average


@dataclass(frozen=True)
class VehicleProfile:
    key: str
    label: str
    fuel: str                       # "diesel" | "petrol" | "cng" | "electric"
    kg_co2_per_km: float | None     # combustion only
    kwh_per_km: float | None        # electric only
    typical_payload_kg: float
    source: str

    @property
    def is_ev(self) -> bool:
        return self.fuel == "electric"

    @property
    def carbon_scale(self) -> float:
        """Multiplier on the GNN's carbon for a combustion vehicle."""
        return (self.kg_co2_per_km or REFERENCE_KG_CO2_PER_KM) / REFERENCE_KG_CO2_PER_KM

    @property
    def energy_scale(self) -> float:
        """Multiplier on the GNN's EV energy for an electric vehicle."""
        return (self.kwh_per_km or REFERENCE_KWH_PER_KM) / REFERENCE_KWH_PER_KM


INDIA_GHG = "India GHG Program road transport factors v1.0 (2015)"
VEHICLE_TYPES = {
    p.key: p for p in (
        VehicleProfile("diesel_ldv", "Diesel light goods vehicle (<3.5 t)", "diesel", 0.3070, None, 750.0, INDIA_GHG),
        VehicleProfile("diesel_mdv", "Diesel medium truck (<12 t)", "diesel", 0.5928, None, 5000.0, INDIA_GHG),
        VehicleProfile("diesel_3w", "Diesel three-wheeler", "diesel", 0.1322, None, 500.0, INDIA_GHG),
        VehicleProfile("cng_3w", "CNG three-wheeler", "cng", 0.10768, None, 500.0, INDIA_GHG),
        VehicleProfile("petrol_2w", "Petrol scooter (<110 cc)", "petrol", 0.0334, None, 20.0, INDIA_GHG),
        VehicleProfile("ev_ldv", "Electric light goods vehicle", "electric", None, REFERENCE_KWH_PER_KM, 750.0,
                       "the dataset's own 0.25 kWh/km; grid CO2 from CEA v20.0"),
    )
}


def profile(vehicle_type: str | None) -> VehicleProfile | None:
    if vehicle_type is None:
        return None
    if vehicle_type not in VEHICLE_TYPES:
        raise ValueError(f"Unknown vehicle_type {vehicle_type!r}; expected one of {sorted(VEHICLE_TYPES)}")
    return VEHICLE_TYPES[vehicle_type]


def vehicle_co2_kg(vehicle_type: str | None, carbon_kg: float, ev_energy_pct: float) -> float:
    """
    CO2 for a route (or segment) driven by this vehicle type, from the GNN's
    reference figures for it. vehicle_type None = the GNN's own reference
    vehicle, carbon_kg unchanged.
    """
    p = profile(vehicle_type)
    if p is None:
        return carbon_kg
    if p.is_ev:
        kwh = ev_energy_pct / 100.0 * EV_REFERENCE_KWH * p.energy_scale
        return kwh * GRID_KG_CO2_PER_KWH
    return carbon_kg * p.carbon_scale


def vehicle_energy_kwh(vehicle_type: str | None, ev_energy_pct: float) -> float:
    """Battery energy for an electric vehicle type (0 for combustion)."""
    p = profile(vehicle_type)
    if p is not None and not p.is_ev:
        return 0.0
    scale = p.energy_scale if p is not None else 1.0
    return ev_energy_pct / 100.0 * EV_REFERENCE_KWH * scale


def costs_for_vehicle(costs_df: pd.DataFrame, vehicle_type: str | None) -> pd.DataFrame:
    """
    The GNN's per-segment costs restated for a vehicle type: carbon_kg
    becomes that vehicle's CO2 per segment, and for an electric type
    ev_energy_pct is scaled to its energy use (still as % of the 50 kWh
    reference battery, so battery checks stay the same). Time and delay are
    unchanged.

    Routing note: for a combustion vehicle this is a uniform scale, and the
    solver's percentile scaling makes routes identical to the reference
    vehicle's. For an electric vehicle the carbon objective becomes its
    grid CO2, which follows energy use.
    """
    p = profile(vehicle_type)
    if p is None:
        return costs_df
    out = costs_df.copy()
    if p.is_ev:
        energy_pct = out["ev_energy_pct"].to_numpy(dtype=float) * p.energy_scale
        out["ev_energy_pct"] = energy_pct
        out["carbon_kg"] = energy_pct / 100.0 * EV_REFERENCE_KWH * GRID_KG_CO2_PER_KWH
    else:
        out["carbon_kg"] = out["carbon_kg"].to_numpy(dtype=float) * p.carbon_scale
    return out


class FleetCosts:
    """
    Per-vehicle-type cost frames and routing solvers for one snapshot, built
    on first use and shared by the agents. A request is routed on the costs
    of the vehicle it was assigned: for an electric vehicle that changes the
    route in some trips (its carbon is grid CO2, which follows energy use);
    for a combustion vehicle it never does (its carbon is a uniform scale of
    the reference), but every carbon figure is then in its own terms.

    Vehicles without a vehicle_type use the GNN's costs unchanged, so a fleet
    with no types behaves exactly as before Phase 5.
    """

    def __init__(self, costs_df: pd.DataFrame, vehicles=None):
        self.base = costs_df
        self.type_by_vehicle = {v.vehicle_id: v.vehicle_type for v in (vehicles or [])}
        self._costs = {None: costs_df}
        self._solvers = {}

    def vehicle_type(self, vehicle_id: str | None) -> str | None:
        return self.type_by_vehicle.get(vehicle_id)

    def costs(self, vehicle_id: str | None = None) -> pd.DataFrame:
        vt = self.vehicle_type(vehicle_id)
        if vt not in self._costs:
            self._costs[vt] = costs_for_vehicle(self.base, vt)
        return self._costs[vt]

    def solver(self, vehicle_id: str | None = None):
        from routing_solver import RoutingSolver  # here, so the catalogue above needs no networkx

        vt = self.vehicle_type(vehicle_id)
        if vt not in self._solvers:
            self._solvers[vt] = RoutingSolver(self.costs(vehicle_id))
        return self._solvers[vt]

