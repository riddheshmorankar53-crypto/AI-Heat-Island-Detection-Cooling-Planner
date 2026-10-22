"""
Spatial data layer for Urban Cooling Planner.
Serves the processed 100m grid from an in-memory GeoPandas frame, so the API runs
on a laptop with no database. (The data pipeline can still export to PostGIS.)
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd

from backend.app.config import settings

# Alert tiers are ranked within hotspots (HVI percentile), so they hold whether the
# grid is synthetic or real satellite data.
ALERT_EXTREME_PCTL = 0.90
ALERT_HIGH_PCTL = 0.50
ALERT_REFUGE_DISTANCE_M = 400.0  # ~5-minute walk
ALERT_VULNERABLE_RESIDENTS = 100  # elderly 60+ and under-5s living in the cell
SEVERITIES = ("moderate", "high", "extreme")


class SpatialDataManager:
    """Loads the 100m grid once and answers grid, summary and alert queries."""

    def __init__(self) -> None:
        self.gdf: Optional[gpd.GeoDataFrame] = None
        self._load_data()

    def _load_data(self) -> None:
        """Load the processed grid, generating it on first run."""
        grid_path = Path(settings.grid_geojson_path)
        if not grid_path.exists():
            from data.pipeline import PuneDataPipeline

            pipeline = PuneDataPipeline()
            self.gdf = pipeline.generate_and_save_grid()
        else:
            self.gdf = gpd.read_file(grid_path)

        print(f"Loaded {len(self.gdf)} 100m grid cells into spatial manager.")

    def get_status(self) -> Dict[str, Any]:
        """Return data store status."""
        source = "unknown"
        if self.gdf is not None and "satellite_source" in self.gdf.columns:
            source = str(self.gdf["satellite_source"].iloc[0])
        return {
            "mode": "GeoPandas (in-memory)",
            "total_cells_cached": len(self.gdf) if self.gdf is not None else 0,
            "crs": str(self.gdf.crs) if self.gdf is not None else "Unknown",
            "satellite_source": source,
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

    def get_alerts(self, limit: int = 20, escalation: int = 0) -> Dict[str, Any]:
        """Rank critical hotspots into heat hazard alerts by vulnerability.

        Args:
            limit: Number of alerts returned (highest HVI first).
            escalation: Severity levels to add (0-2), e.g. from a heatwave forecast.
        """
        if self.gdf is None:
            return {"total_alerts": 0, "counts": {}, "alerts": []}

        hotspots = self.gdf[self.gdf["is_critical_hotspot"]].sort_values(
            "heat_vulnerability_index", ascending=False
        )
        hvi_pctl = hotspots["heat_vulnerability_index"].rank(pct=True)
        alerts = []
        for idx, c in hotspots.iterrows():
            level = int(hvi_pctl[idx] >= ALERT_HIGH_PCTL) + int(hvi_pctl[idx] >= ALERT_EXTREME_PCTL)
            severity = SEVERITIES[min(level + escalation, 2)]

            actions = ["Open water/misting points during 12:00-16:00"]
            if c["dist_cooling_facility_m"] > ALERT_REFUGE_DISTANCE_M:
                actions.append(
                    f"No cooling refuge within {int(c['dist_cooling_facility_m'])} m: deploy temporary shelter"
                )
            vulnerable = c["population"] * c["vuln_population_ratio"]
            if vulnerable >= ALERT_VULNERABLE_RESIDENTS:
                actions.append(
                    f"Door-to-door checks on ~{int(vulnerable)} elderly residents and young children"
                )

            alerts.append(
                {
                    "cell_id": c["cell_id"],
                    "severity": severity,
                    "lon": float(c["centroid_lon"]),
                    "lat": float(c["centroid_lat"]),
                    "lst_celsius": float(c["lst_celsius"]),
                    "lst_anomaly_celsius": float(c["lst_anomaly_celsius"]),
                    "heat_vulnerability_index": float(c["heat_vulnerability_index"]),
                    "population": int(c["population"]),
                    "dist_cooling_facility_m": float(c["dist_cooling_facility_m"]),
                    "actions": actions,
                }
            )

        counts = {s: sum(a["severity"] == s for a in alerts) for s in reversed(SEVERITIES)}
        return {"total_alerts": len(alerts), "counts": counts, "alerts": alerts[:limit]}

    def get_raw_gdf(self) -> gpd.GeoDataFrame:
        """Return the underlying GeoPandas GeoDataFrame."""
        return self.gdf


# Singleton spatial data manager instance
spatial_db = SpatialDataManager()
