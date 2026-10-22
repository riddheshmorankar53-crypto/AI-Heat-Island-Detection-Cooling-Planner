"""Integration tests for FastAPI REST endpoints."""

from fastapi.testclient import TestClient

from backend.app import forecast
from backend.app.main import app

client = TestClient(app)


def test_health_check():
    """Verify system health endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "database" in data


def test_get_aoi():
    """Verify AOI metadata endpoint."""
    response = client.get("/api/v1/aoi")
    assert response.status_code == 200
    data = response.json()
    assert data["city"] == "Pune"
    assert "bbox" in data
    assert len(data["landmarks"]) > 0


def test_get_hotspots_summary():
    """Verify hotspots summary endpoint."""
    response = client.get("/api/v1/hotspots/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_cells" in data
    assert data["total_cells"] > 0
    assert "critical_hotspot_cells" in data
    assert "max_lst_celsius" in data


def test_get_grid_filtered():
    """Verify grid GeoJSON retrieval with limit."""
    response = client.get("/api/v1/grid?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 10


def test_heat_alerts(monkeypatch):
    """Verify alerts are hotspot cells, ranked by HVI, with consistent counts."""
    monkeypatch.setattr(forecast, "fetch_forecast", lambda: None)  # no network in tests
    response = client.get("/api/v1/alerts?limit=5")
    assert response.status_code == 200
    data = response.json()
    summary = client.get("/api/v1/hotspots/summary").json()
    assert data["total_alerts"] == summary["critical_hotspot_cells"]
    assert sum(data["counts"].values()) == data["total_alerts"]
    alerts = data["alerts"]
    assert 0 < len(alerts) <= 5
    hvis = [a["heat_vulnerability_index"] for a in alerts]
    assert hvis == sorted(hvis, reverse=True)
    assert all(a["severity"] in ("extreme", "high", "moderate") and a["actions"] for a in alerts)


def test_heatwave_forecast_escalates_alerts(monkeypatch):
    """A forecast heatwave raises every alert's severity; a severe one makes all extreme."""

    def fake(tmax):
        return {"source": "test", "days": [{"date": "2026-05-01", "tmax_celsius": tmax}]}

    def counts_for(tmax):
        monkeypatch.setattr(forecast, "fetch_forecast", lambda: fake(tmax))
        data = client.get("/api/v1/alerts").json()
        return data["counts"], data["escalation_levels"]

    calm, esc0 = counts_for(35.0)
    heatwave, esc1 = counts_for(41.0)
    severe, esc2 = counts_for(46.0)
    assert (esc0, esc1, esc2) == (0, 1, 2)
    assert calm["extreme"] > 0 and calm["high"] > 0 and calm["moderate"] > 0
    assert heatwave["moderate"] == 0
    assert heatwave["extreme"] == calm["extreme"] + calm["high"]
    assert severe["extreme"] == sum(calm.values())


def test_interventions_catalog():
    """Verify catalog endpoint."""
    response = client.get("/api/v1/interventions/catalog")
    assert response.status_code == 200
    data = response.json()
    assert "trees" in data
    assert "cooling_centers" in data


def test_interventions_optimize():
    """Verify optimization endpoint."""
    payload = {
        "budget_inr": 800000.0,
        "enabled_types": ["trees", "water_points", "shaded_stops"],
        "max_interventions_per_cell": 1,
    }
    response = client.post("/api/v1/interventions/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("OPTIMAL", "FEASIBLE")
    assert data["summary"]["total_cost_inr"] <= 800000.0
    assert len(data["features"]) > 0


def test_heat_safe_routing():
    """Verify routing endpoint."""
    payload = {
        "origin": {"lon": 73.8553, "lat": 18.5196},
        "destination": {"lon": 73.8744, "lat": 18.5284},
        "heat_avoidance_weight": 2.0,
    }
    response = client.post("/api/v1/routing/heat-safe", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fastest_route" in data
    assert "heat_safe_route" in data
    assert "comparison" in data


def test_ml_metrics():
    """Verify model metrics endpoint returns genuine saved numbers."""
    response = client.get("/api/v1/ml/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "metadata" in data
    assert "models" in data
    assert "lightgbm_lst_regressor" in data["models"]
