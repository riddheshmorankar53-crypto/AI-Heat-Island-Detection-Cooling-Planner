"""
Heat-Safe Pedestrian Walking Route Endpoints.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from backend.app.schemas.routing import RouteRequest, RouteResponse
from ml.routing.heat_safe_router import HeatSafeRouter

router = APIRouter(prefix="/api/v1/routing", tags=["Heat-Safe Routing"])

# Shared router instance
heat_router = HeatSafeRouter()


@router.get("/landmarks")
def get_routing_landmarks() -> List[Dict[str, Any]]:
    """Retrieve preset Pune landmarks for rapid origin/destination selection."""
    landmarks = []
    for nid, data in heat_router.nodes.items():
        landmarks.append(
            {
                "id": nid,
                "name": nid.replace("_", " ").title(),
                "lon": data["lon"],
                "lat": data["lat"],
                "ambient_lst": data.get("ambient_lst", 36.0),
                "type": data.get("type", "standard"),
            }
        )
    return sorted(landmarks, key=lambda x: x["name"])


@router.post("/heat-safe", response_model=RouteResponse)
def compute_route(req: RouteRequest) -> RouteResponse:
    """
    Compute dual pedestrian routes: standard fastest vs heat-safe route
    favoring tree-lined streets, riverwalk greenways, and shaded corridors.
    """
    try:
        result = heat_router.route(
            origin_lon=req.origin.lon,
            origin_lat=req.origin.lat,
            dest_lon=req.destination.lon,
            dest_lat=req.destination.lat,
            heat_avoidance_weight=req.heat_avoidance_weight,
        )
        return RouteResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing error: {str(e)}")
