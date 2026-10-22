"""Schemas for pedestrian heat-safe walking routes."""

from typing import Any, Dict

from pydantic import BaseModel, Field


class Coordinates(BaseModel):
    lon: float
    lat: float


class RouteRequest(BaseModel):
    origin: Coordinates = Field(..., description="Origin coordinates (lon, lat)")
    destination: Coordinates = Field(..., description="Destination coordinates (lon, lat)")
    heat_avoidance_weight: float = Field(
        default=2.0, ge=0.0, le=5.0, description="Heat avoidance sensitivity parameter (gamma)"
    )


class RouteComparison(BaseModel):
    heat_stress_reduction_pct: float
    extra_distance_meters: float
    extra_walk_minutes: float
    shade_increase_pct: float


class RouteResponse(BaseModel):
    origin: Dict[str, Any]
    destination: Dict[str, Any]
    fastest_route: Dict[str, Any]
    heat_safe_route: Dict[str, Any]
    comparison: RouteComparison
