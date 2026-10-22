"""Schemas for cooling intervention optimization."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class InterventionOptimizeRequest(BaseModel):
    budget_inr: float = Field(
        default=1500000.0, ge=50000.0, le=50000000.0, description="Municipal budget in INR"
    )
    enabled_types: Optional[List[str]] = Field(
        default=["trees", "cool_roofs", "water_points", "shaded_stops", "cooling_centers"],
        description="Subset of enabled intervention categories",
    )
    max_interventions_per_cell: int = Field(default=1, ge=1, le=5)
    priority_hotspots_only: bool = Field(
        default=False, description="Restrict candidate sites to critical hotspots"
    )


class InterventionSummary(BaseModel):
    budget_allocated_inr: float
    total_cost_inr: float
    budget_utilization_pct: float
    interventions_count: int
    type_breakdown: Dict[str, int]
    avg_cooling_impact_celsius: float
    total_population_protected: int


class InterventionOptimizeResponse(BaseModel):
    status: str
    summary: InterventionSummary
    features: List[Dict[str, Any]]
