"""Schemas for 100m grid and hotspot statistics."""

from typing import Any, Dict, List

from pydantic import BaseModel


class HotspotsSummaryResponse(BaseModel):
    total_cells: int
    critical_hotspot_cells: int
    hotspot_percentage: float
    max_lst_celsius: float
    min_lst_celsius: float
    mean_lst_celsius: float
    total_population: int
    vulnerable_population_at_risk: int
    avg_heat_vulnerability_index: float


class AOIResponse(BaseModel):
    name: str
    city: str
    state: str
    country: str
    bbox: Dict[str, float]
    center: Dict[str, float]
    landmarks: List[Dict[str, Any]]
