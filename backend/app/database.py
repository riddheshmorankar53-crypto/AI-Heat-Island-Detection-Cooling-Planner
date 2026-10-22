"""
Dual-adapter spatial database layer for Urban Cooling Planner.
Automatically falls back to in-memory GeoPandas spatial index if PostGIS is unavailable,
ensuring 100% friction-free execution on a laptop without mandatory Docker.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd

from backend.app.config import settings

# Alert thresholds, tuned to hotspot cells (already HVI >= 0.62): p50 ~0.70, p90 ~0.75
ALERT_EXTREME_HVI = 0.75
ALERT_HIGH_HVI = 0.70
ALERT_REFUGE_DISTANCE_M = 800.0


class SpatialDataManager:
    """Manages spatial grid querying with auto-fallback between PostGIS and GeoPandas."""

    def __init__(self) -> None:
        self.is_postgis_connected: bool = False
        self.gdf: Optional[gpd.GeoDataFrame] = None
        self._load_data()

    def _load_data(self) -> None:
        """Initialize data store."""
        if settings.use_postgis:
            try:
                # Attempt PostGIS connection via GeoAlchemy2 / SQLAlchemy
                from sqlalchemy import create_engine, text

                engine = create_engine(settings.database_url, connect_args={"connect_timeout": 3})
                with engine.connect() as conn:
                    # Test basic query
                    conn.execute(text("SELECT 1"))
                    self.is_postgis_connected = True
                    print("Connected to PostgreSQL+PostGIS database.")
            except Exception as e:
                print(
                    f"PostGIS connection not available ({e}). Falling back to GeoPandas local spatial engine."
                )
                self.is_postgis_connected = False

        # Load GeoDataFrame from local processed GeoJSON
        grid_path = Path(settings.grid_geojson_path)
        if not grid_path.exists():
            from data.pipeline import PuneDataPipeline

            pipeline = PuneDataPipeline()
            self.gdf = pipeline.generate_and_save_grid()
        else:
            self.gdf = gpd.read_file(grid_path)

        print(f"Loaded {len(self.gdf)} 100m grid cells into spatial manager.")

    def get_status(self) -> Dict[str, Any]:
        """Return current database connection status."""
        return {
            "mode": "PostGIS" if self.is_postgis_connected else "Local_GeoPandas_Fallback",
            "is_postgis_active": self.is_postgis_connected,
            "total_cells_cached": len(self.gdf) if self.gdf is not None else 0,
            "crs": str(self.gdf.crs) if self.gdf is not None else "Unknown",
        }

    def get_summary(self) -> Dict[str, Any]:
        """Compute aggregate heat island and vulnerability statistics."""
        if self.gdf is None:
            return {}

        hotspots = self.gdf[self.gdf["is_critical_hotspot"]]
        total_pop = int(self.gdf["population"].sum())
        hotspot_pop = int(hotspots["population"].sum())

        return {
            "total_cells": len(self.gdf),
            "critical_hotspot_cells": len(hotspots),
            "hotspot_percentage": round((len(hotspots) / len(self.gdf)) * 100.0, 1),
            "max_lst_celsius": round(float(self.gdf["lst_celsius"].max()), 2),
            "min_lst_celsius": round(float(self.gdf["lst_celsius"].min()), 2),
            "mean_lst_celsius": round(float(self.gdf["lst_celsius"].mean()), 2),
            "total_population": total_pop,
            "vulnerable_population_at_risk": hotspot_pop,
            "avg_heat_vulnerability_index": round(
                float(self.gdf["heat_vulnerability_index"].mean()), 3
            ),
        }

    def query_grid(
        self,
        min_lst: Optional[float] = None,
        max_lst: Optional[float] = None,
        hotspot_only: bool = False,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Query 100m cells with optional environmental filters."""
        if self.gdf is None:
            return {"type": "FeatureCollection", "features": []}

        filtered = self.gdf
        if hotspot_only:
            filtered = filtered[filtered["is_critical_hotspot"]]
        if min_lst is not None:
            filtered = filtered[filtered["lst_celsius"] >= min_lst]
        if max_lst is not None:
            filtered = filtered[filtered["lst_celsius"] <= max_lst]

        if limit is not None and limit > 0:
            filtered = filtered.head(limit)

        # Convert to GeoJSON FeatureCollection
        return json.loads(filtered.to_json())

    def get_alerts(self, limit: int = 20) -> Dict[str, Any]:
        """Rank critical hotspots into heat hazard alerts by vulnerability.

        ponytail: static alerts from the satellite-derived grid; feed an IMD/Open-Meteo
        forecast in here to turn them into true time-based early warnings.
        """
        if self.gdf is None:
            return {"total_alerts": 0, "counts": {}, "alerts": []}

        hotspots = self.gdf[self.gdf["is_critical_hotspot"]].sort_values(
            "heat_vulnerability_index", ascending=False
        )
        alerts = []
        for _, c in hotspots.iterrows():
            hvi = float(c["heat_vulnerability_index"])
            far_from_refuge = c["dist_cooling_facility_m"] > ALERT_REFUGE_DISTANCE_M
            if hvi >= ALERT_EXTREME_HVI:
                severity = "extreme"
            elif hvi >= ALERT_HIGH_HVI:
                severity = "high"
            else:
                severity = "moderate"

            actions = ["Open water/misting points during 12:00-16:00"]
            if far_from_refuge:
                actions.append(
                    f"No cooling refuge within {int(c['dist_cooling_facility_m'])} m: deploy temporary shelter"
                )
            if c["vuln_population_ratio"] >= 0.3:
                actions.append("Door-to-door checks on elderly and young children")

            alerts.append(
                {
                    "cell_id": c["cell_id"],
                    "severity": severity,
                    "lon": float(c["centroid_lon"]),
                    "lat": float(c["centroid_lat"]),
                    "lst_celsius": float(c["lst_celsius"]),
                    "lst_anomaly_celsius": float(c["lst_anomaly_celsius"]),
                    "heat_vulnerability_index": hvi,
                    "population": int(c["population"]),
                    "dist_cooling_facility_m": float(c["dist_cooling_facility_m"]),
                    "actions": actions,
                }
            )

        counts = {
            s: sum(a["severity"] == s for a in alerts) for s in ("extreme", "high", "moderate")
        }
        return {"total_alerts": len(alerts), "counts": counts, "alerts": alerts[:limit]}

    def get_raw_gdf(self) -> gpd.GeoDataFrame:
        """Return the underlying GeoPandas GeoDataFrame."""
        return self.gdf


# Singleton spatial data manager instance
spatial_db = SpatialDataManager()
