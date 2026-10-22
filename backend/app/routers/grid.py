"""
Grid, Hotspots, and AOI Endpoints.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from fastapi import APIRouter, Query

from backend.app import forecast
from backend.app.config import settings
from backend.app.database import spatial_db
from backend.app.schemas.grid import AOIResponse, HotspotsSummaryResponse

router = APIRouter(prefix="/api/v1", tags=["Grid & Heat Islands"])


@router.get("/grid")
def get_grid(
    min_lst: Optional[float] = Query(None, description="Filter minimum LST in Celsius"),
    max_lst: Optional[float] = Query(None, description="Filter maximum LST in Celsius"),
    hotspot_only: bool = Query(False, description="Filter to critical hotspots only"),
    limit: Optional[int] = Query(None, description="Limit number of returned cells"),
) -> Dict[str, Any]:
    """Retrieve 100m grid cells as GeoJSON with environmental features."""
    return spatial_db.query_grid(
        min_lst=min_lst,
        max_lst=max_lst,
        hotspot_only=hotspot_only,
        limit=limit,
    )


@router.get("/hotspots/summary", response_model=HotspotsSummaryResponse)
def get_hotspots_summary() -> HotspotsSummaryResponse:
    """Get aggregated summary of urban heat island intensity and population vulnerability."""
    summary_data = spatial_db.get_summary()
    return HotspotsSummaryResponse(**summary_data)


@router.get("/alerts")
def get_heat_alerts(
    limit: int = Query(20, ge=1, le=500, description="Max alerts returned (highest risk first)"),
) -> Dict[str, Any]:
    """Heat hazard alerts for critical hotspots, escalated when a heatwave is forecast."""
    fc = forecast.fetch_forecast()
    escalation = forecast.escalation_for(fc)
    return {
        **spatial_db.get_alerts(limit=limit, escalation=escalation),
        "forecast": fc,
        "escalation_levels": escalation,
    }


@router.get("/aoi", response_model=AOIResponse)
def get_aoi_metadata() -> AOIResponse:
    """Get Pune Pilot AOI bounding box, center, and key landmark anchors."""
    aoi_path = Path(settings.aoi_yaml_path)
    with open(aoi_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)["aoi"]

    return AOIResponse(
        name=data["name"],
        city=data["city"],
        state=data["state"],
        country=data["country"],
        bbox=data["bbox"],
        center=data["center"],
        landmarks=data["landmarks"],
    )
