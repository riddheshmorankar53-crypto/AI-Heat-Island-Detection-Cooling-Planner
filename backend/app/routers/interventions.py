"""
Cooling Intervention Optimization Endpoints.
"""

import threading
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from backend.app.database import spatial_db
from backend.app.schemas.interventions import (
    InterventionOptimizeRequest,
    InterventionOptimizeResponse,
)
from ml.optimizer.intervention_solver import INTERVENTION_CATALOG, CoolingInterventionOptimizer

router = APIRouter(prefix="/api/v1/interventions", tags=["Cooling Interventions"])
optimizer = CoolingInterventionOptimizer()
# Each solve holds ~100 MB; concurrent solves would exceed a 512 MB free-tier instance.
# ponytail: global lock (requests queue ~1s each); use a worker pool if traffic grows.
_solver_lock = threading.Lock()


@router.get("/catalog")
def get_catalog() -> Dict[str, Any]:
    """Retrieve catalog of available cooling interventions with unit costs and expected delta T."""
    catalog_clean = {}
    for k, v in INTERVENTION_CATALOG.items():
        catalog_clean[k] = {
            "name": v["name"],
            "cost_inr": v["cost_inr"],
            "delta_celsius": v["delta_celsius"],
            "basis": v["basis"],
            "influence_radius_m": v["influence_radius_m"],
            "color": v["color"],
            "icon": v["icon"],
        }
    return catalog_clean


@router.post("/optimize", response_model=InterventionOptimizeResponse)
def optimize_interventions(req: InterventionOptimizeRequest) -> InterventionOptimizeResponse:
    """
    Run Google OR-Tools MILP optimizer to allocate cooling interventions within municipal budget.
    """
    raw_gdf = spatial_db.get_raw_gdf()
    if raw_gdf is None or len(raw_gdf) == 0:
        raise HTTPException(status_code=500, detail="Spatial grid data not loaded.")

    try:
        with _solver_lock:
            result = optimizer.optimize(
                grid_gdf=raw_gdf,
                budget_inr=req.budget_inr,
                enabled_types=req.enabled_types,
                max_interventions_per_cell=req.max_interventions_per_cell,
                priority_hotspots_only=req.priority_hotspots_only,
            )
        return InterventionOptimizeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization solver error: {str(e)}")
