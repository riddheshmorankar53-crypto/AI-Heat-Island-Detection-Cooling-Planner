"""
Smoke tests verifying monorepo scaffolding, configuration integrity, and module stubs.
"""

from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent


def test_city_config_loaded():
    """Verify city.yaml contains required Pune pilot specifications."""
    city_yaml_path = BASE_DIR / "config" / "city.yaml"
    assert city_yaml_path.exists(), "config/city.yaml must exist"

    with open(city_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)["city"]

    assert data["name"] == "Pune"
    assert data["crs"]["projected"] == "EPSG:32643"
    assert data["grid_size_meters"] == 100.0

    bbox = data["bounding_box"]
    assert bbox["min_latitude"] < bbox["max_latitude"]
    assert bbox["min_longitude"] < bbox["max_longitude"]

    # Verify summer 2023-2025 date ranges
    seasons = [item["season"] for item in data["summer_date_ranges"]]
    assert "Summer 2023" in seasons
    assert "Summer 2024" in seasons
    assert "Summer 2025" in seasons


def test_master_config_loaded():
    """Verify config.yaml contains required application settings."""
    config_yaml_path = BASE_DIR / "config" / "config.yaml"
    assert config_yaml_path.exists(), "config/config.yaml must exist"

    with open(config_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["project"]["seed"] == 42
    assert "paths" in cfg
    assert "optimizer" in cfg


def test_pyproject_toml_exists():
    """Verify pyproject.toml exists and declares dependencies."""
    pyproject_path = BASE_DIR / "pyproject.toml"
    assert pyproject_path.exists()
    content = pyproject_path.read_text(encoding="utf-8")
    assert "urban-cooling-planner" in content
    assert "fastapi" in content
    assert "geopandas" in content


def test_module_stubs_importable():
    """Verify core module entrypoints can be imported without errors."""
    import backend.app.main
    import data.pipeline
    import ml.models.baseline
    import ml.optimizer.intervention_solver
    import ml.routing.heat_safe_router

    assert backend.app.main.app is not None
    assert data.pipeline.PuneDataPipeline is not None
    assert ml.optimizer.intervention_solver.CoolingInterventionOptimizer is not None
    assert ml.routing.heat_safe_router.HeatSafeRouter is not None
