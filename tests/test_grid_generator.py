"""Unit tests for 100m grid generator."""

import geopandas as gpd
import pytest

from data.synthetic.generator import PuneSyntheticGridGenerator


@pytest.fixture
def pune_bbox():
    return {
        "min_lat": 18.5000,
        "min_lon": 73.8400,
        "max_lat": 18.5200,
        "max_lon": 73.8600,
    }


def test_grid_generator_dimensions(pune_bbox):
    """Verify grid generation produces valid polygons in EPSG:4326."""
    gen = PuneSyntheticGridGenerator(bbox=pune_bbox, resolution_meters=100.0, seed=42)
    gdf = gen.generate()

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) > 0
    assert gdf.crs == "EPSG:4326"
    assert "geometry" in gdf.columns
    assert "lst_celsius" in gdf.columns
    assert "is_synthetic" in gdf.columns
    assert gdf["is_synthetic"].all()


def test_environmental_ranges(pune_bbox):
    """Verify physical ranges of microclimate and satellite indices."""
    gen = PuneSyntheticGridGenerator(bbox=pune_bbox, resolution_meters=100.0, seed=42)
    gdf = gen.generate()

    assert gdf["ndvi"].between(-0.2, 1.0).all()
    assert gdf["ndbi"].between(-0.5, 1.0).all()
    assert gdf["lst_celsius"].between(25.0, 50.0).all()
    assert gdf["heat_vulnerability_index"].between(0.0, 1.0).all()
    assert gdf["geometry"].is_valid.all()
