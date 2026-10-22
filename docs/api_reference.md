# REST API Reference: Urban Cooling Planner

Base URL: `http://localhost:8000/api/v1`
Interactive Swagger / OpenAPI Docs: `http://localhost:8000/docs`

---

## 1. System & Health

### `GET /health`
Returns service health and spatial database connection status.

**Response `200 OK`**:
```json
{
  "status": "healthy",
  "app": "Urban Cooling Planner API",
  "version": "1.0.0",
  "environment": "development",
  "pilot_city": "Pune, India",
  "database": {
    "mode": "GeoPandas (in-memory)",
    "total_cells_cached": 3904,
    "crs": "EPSG:4326",
    "satellite_source": "landsat-c2-l2 + sentinel-2-l2a + cop-dem-glo-30"
  }
}
```

---

## 2. GIS Grid & Hotspots

### `GET /aoi`
Returns Pune Pilot AOI metadata, bounding box coordinates, and landmark hubs.

### `GET /grid`
Returns 100m grid cells as a GeoJSON FeatureCollection.

**Query Parameters**:
- `min_lst` (float, optional): Minimum Land Surface Temperature in Celsius.
- `max_lst` (float, optional): Maximum Land Surface Temperature in Celsius.
- `hotspot_only` (bool, optional): Filter for critical heat hotspots only.
- `limit` (int, optional): Restrict number of cells returned.

**Response `200 OK`**: GeoJSON FeatureCollection with cell properties (`cell_id`, `lst_celsius`, `ndvi`, `ndbi`, `tree_canopy_pct`, `impervious_pct`, `population`, `heat_vulnerability_index`, `is_critical_hotspot`).

### `GET /hotspots/summary`
Aggregated statistics on urban heat island intensity and vulnerable population at risk.

**Response `200 OK`**:
```json
{
  "total_cells": 3904,
  "critical_hotspot_cells": 534,
  "hotspot_percentage": 13.7,
  "max_lst_celsius": 50.37,
  "min_lst_celsius": 33.67,
  "mean_lst_celsius": 41.19,
  "total_population": 1171850,
  "vulnerable_population_at_risk": 215207,
  "avg_heat_vulnerability_index": 0.537
}
```

### `GET /alerts`
Heat hazard alerts for every critical hotspot cell, ranked by Heat Vulnerability Index.
Base severity is ranked within hotspots: top 10% HVI `extreme`, next 40% `high`, the rest `moderate`.
A 3-day Open-Meteo forecast for Pune then escalates every alert by `escalation_levels`: +1 if any day's max reaches 40°C, +2 at 45°C (capped at `extreme`). `forecast` is `null` when Open-Meteo is unreachable, and severity falls back to the base tiers.
Cells more than 400 m (a 5-minute walk) from a cooling facility get a "deploy temporary shelter" action, and cells holding 100+ elderly residents and young children get a door-to-door check action.

**Query Parameters**:
- `limit` (int, default 20, 1–500): Number of alerts returned (highest risk first). `total_alerts` and `counts` always cover all hotspots.

**Response `200 OK`**:
```json
{
  "total_alerts": 271,
  "counts": { "extreme": 28, "high": 108, "moderate": 135 },
  "alerts": [
    {
      "cell_id": "PUNE_100M_01482",
      "severity": "extreme",
      "lon": 73.8481, "lat": 18.5069,
      "lst_celsius": 47.34, "lst_anomaly_celsius": 6.15,
      "heat_vulnerability_index": 0.75,
      "population": 286,
      "dist_cooling_facility_m": 2293.0,
      "actions": ["Open water/misting points during 12:00-16:00", "..."]
    }
  ],
  "forecast": {
    "source": "open-meteo",
    "days": [{ "date": "2026-10-22", "tmax_celsius": 33.5 }, "..."]
  },
  "escalation_levels": 0
}
```

---

## 3. Cooling Intervention Optimization

### `GET /interventions/catalog`
Catalog of cooling interventions with unit costs, expected cooling deltas, and a `basis` string saying whether each delta was measured on the Pune grid (trees) or is a literature assumption.

### `POST /interventions/optimize`
Solves Google OR-Tools Mixed-Integer Linear Program for intervention placement under municipal budget.

**Request Body**:
```json
{
  "budget_inr": 1500000.0,
  "enabled_types": ["trees", "cool_roofs", "water_points", "shaded_stops", "cooling_centers"],
  "max_interventions_per_cell": 1,
  "priority_hotspots_only": false
}
```

**Response `200 OK`**:
```json
{
  "status": "OPTIMAL",
  "summary": {
    "budget_allocated_inr": 1500000.0,
    "total_cost_inr": 1495000.0,
    "budget_utilization_pct": 99.7,
    "interventions_count": 51,
    "type_breakdown": {
      "trees": 45,
      "cool_roofs": 0,
      "water_points": 0,
      "shaded_stops": 4,
      "cooling_centers": 2
    },
    "avg_cooling_impact_celsius": 1.66,
    "total_population_protected": 18888
  },
  "features": [ ... ]
}
```

---

## 4. Heat-Safe Pedestrian Routing

### `GET /routing/landmarks`
Returns available landmark hubs across Central Pune (Shaniwar Wada, Pune Station, FC Road, Swargate, etc.).

### `POST /routing/heat-safe`
Calculates dual routes (fastest vs heat-safe) using NetworkX thermal cost weighting.

**Request Body**:
```json
{
  "origin": { "lon": 73.8553, "lat": 18.5196 },
  "destination": { "lon": 73.8744, "lat": 18.5284 },
  "heat_avoidance_weight": 2.5
}
```

**Response `200 OK`**:
```json
{
  "origin": { "node_id": "SHANIWAR_WADA", "lon": 73.8553, "lat": 18.5196 },
  "destination": { "node_id": "PUNE_STATION", "lon": 73.8744, "lat": 18.5284 },
  "fastest_route": {
    "type": "Feature",
    "geometry": { "type": "LineString", "coordinates": [...] },
    "properties": { "distance_meters": 2410.4, "walk_time_minutes": 32.1, "shade_percentage": 11.7 }
  },
  "heat_safe_route": {
    "type": "Feature",
    "geometry": { "type": "LineString", "coordinates": [...] },
    "properties": { "distance_meters": 2750.0, "walk_time_minutes": 36.6, "shade_percentage": 38.5 }
  },
  "comparison": {
    "heat_stress_reduction_pct": 12.5,
    "extra_distance_meters": 339.6,
    "extra_walk_minutes": 4.5,
    "shade_increase_pct": 26.8
  }
}
```

---

## 5. Model Metrics & Transparency

### `GET /ml/metrics`
Returns genuine test-set evaluation figures ($R^2$, RMSE, MAE, Confusion Matrix, and Feature Importances).
