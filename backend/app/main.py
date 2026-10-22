"""
Main FastAPI application entrypoint for Urban Cooling Planner.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import settings
from backend.app.database import spatial_db
from backend.app.routers import grid, interventions, metrics, routing

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="AI + GIS + Graph ML system detecting urban heat hotspots, optimizing cooling interventions, and providing heat-safe pedestrian routing for Pune, India.",
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(grid.router)
app.include_router(interventions.router)
app.include_router(routing.router)
app.include_router(metrics.router)


@app.get("/api/v1/health", tags=["System"])
def health_check():
    """System health check and database status."""
    db_status = spatial_db.get_status()
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "pilot_city": "Pune, India",
        "database": db_status,
    }


# Serve the built dashboard from the same origin (single-container deploy). Mounted last
# so every /api route above takes precedence.
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="dashboard")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
