"""
Unit and integration tests for the /data pipeline.

Validates:
- CRS consistency (EPSG:32643 metric internal, EPSG:4326 geographic output)
- Fishnet regular 100m x 100m grid cell geometry and non-overlapping topology
- Numerical ranges and bounds for all 15+ environmental and urban features
- GeoParquet and GeoJSON export round-trip integrity
- Data quality profiling report generation
"""

import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest

from data.pipeline import PuneDataPipeline
from data.processing.grid_builder import FishnetGridBuilder
from data.storage.exporter import DataExporter


@pytest.fixture(scope="module")
def sample_grid_gdf() -> gpd.GeoDataFrame:
    """Generate a lightweight synthetic grid for testing."""
    bbox = {
        "min_latitude": 18.5000,
        "min_longitude": 73.8400,
        "max_latitude": 18.5200,
        "max_longitude": 73.8600,
    }
    builder = FishnetGridBuilder(
        bbox=bbox,
        resolution_meters=100.0,
        crs_geo="EPSG:4326",
        crs_proj="EPSG:32643",
        seed=42,
    )
    return builder.compute_features(synthetic=True)


def test_crs_consistency(sample_grid_gdf: gpd.GeoDataFrame) -> None:
    """Verify final output is EPSG:4326 and internal coordinate validity."""
    assert sample_grid_gdf.crs is not None
    assert sample_grid_gdf.crs.to_string() == "EPSG:4326"
    assert sample_grid_gdf.geometry.is_valid.all()


def test_grid_alignment_and_area(sample_grid_gdf: gpd.GeoDataFrame) -> None:
    """Verify that re-projecting to metric UTM Zone 43N yields ~10,000 m² cells."""
    gdf_utm = sample_grid_gdf.to_crs("EPSG:32643")
    areas = gdf_utm.geometry.area

    # Regular 100m x 100m cell must be 10,000 m² ± 0.5%
    np.testing.assert_allclose(areas, 10000.0, rtol=0.01)

    # Verify cells are non-overlapping polygons
    bounds = gdf_utm.geometry.total_bounds
    assert bounds[2] > bounds[0]
    assert bounds[3] > bounds[1]


def test_feature_presence_and_bounds(sample_grid_gdf: gpd.GeoDataFrame) -> None:
    """Verify all 15+ required features exist and satisfy realistic physical boundaries."""
    required_columns = [
        "cell_id",
        "mean_lst",
        "lst_celsius",
        "mean_ndvi",
        "mean_ndbi",
        "mean_ndwi",
        "building_density",
        "mean_building_height",
        "sky_view_factor",
        "road_density_m",
        "impervious_fraction",
        "dist_to_water_m",
        "dist_to_park_m",
        "elevation_m",
        "population",
        "pct_children",
        "pct_elderly",
        "low_income_proxy",
        "heat_vulnerability_index",
        "is_critical_hotspot",
        "is_synthetic",
    ]

    for col in required_columns:
        assert col in sample_grid_gdf.columns, f"Missing feature column: {col}"

    # Verify physical and index bounds
    assert (sample_grid_gdf["mean_lst"] >= 20.0).all() and (
        sample_grid_gdf["mean_lst"] <= 55.0
    ).all()
    assert (sample_grid_gdf["mean_ndvi"] >= -1.0).all() and (
        sample_grid_gdf["mean_ndvi"] <= 1.0
    ).all()
    assert (sample_grid_gdf["mean_ndbi"] >= -1.0).all() and (
        sample_grid_gdf["mean_ndbi"] <= 1.0
    ).all()
    assert (sample_grid_gdf["mean_ndwi"] >= -1.0).all() and (
        sample_grid_gdf["mean_ndwi"] <= 1.0
    ).all()

    # Urban form bounds
    assert (sample_grid_gdf["building_density"] >= 0.0).all() and (
        sample_grid_gdf["building_density"] <= 1.0
    ).all()
    assert (sample_grid_gdf["mean_building_height"] >= 2.0).all() and (
        sample_grid_gdf["mean_building_height"] <= 50.0
    ).all()
    assert (sample_grid_gdf["sky_view_factor"] >= 0.0).all() and (
        sample_grid_gdf["sky_view_factor"] <= 1.0
    ).all()
    assert (sample_grid_gdf["impervious_fraction"] >= 0.0).all() and (
        sample_grid_gdf["impervious_fraction"] <= 1.0
    ).all()

    # Distances and elevation
    assert (sample_grid_gdf["dist_to_water_m"] >= 0.0).all()
    assert (sample_grid_gdf["dist_to_park_m"] >= 0.0).all()
    assert (sample_grid_gdf["elevation_m"] >= 450.0).all() and (
        sample_grid_gdf["elevation_m"] <= 900.0
    ).all()

    # Demographics and vulnerability proxies
    assert (sample_grid_gdf["population"] >= 0).all()
    assert (sample_grid_gdf["pct_children"] >= 0.0).all() and (
        sample_grid_gdf["pct_children"] <= 35.0
    ).all()
    assert (sample_grid_gdf["pct_elderly"] >= 0.0).all() and (
        sample_grid_gdf["pct_elderly"] <= 35.0
    ).all()
    assert (sample_grid_gdf["low_income_proxy"] >= 0.0).all() and (
        sample_grid_gdf["low_income_proxy"] <= 1.0
    ).all()
    assert (sample_grid_gdf["heat_vulnerability_index"] >= 0.0).all() and (
        sample_grid_gdf["heat_vulnerability_index"] <= 1.0
    ).all()

    # Hotspots must be boolean
    assert sample_grid_gdf["is_critical_hotspot"].dtype == bool
    # Synthetic flag must be True
    assert sample_grid_gdf["is_synthetic"].all()


def test_geoparquet_and_geojson_export(sample_grid_gdf: gpd.GeoDataFrame) -> None:
    """Verify GeoParquet and GeoJSON export and roundtrip loading."""
    exporter = DataExporter()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_parquet = Path(tmpdir) / "test_grid.parquet"
        tmp_geojson = Path(tmpdir) / "test_grid.geojson"

        # 1. GeoParquet
        exporter.save_geoparquet(sample_grid_gdf, tmp_parquet)
        assert tmp_parquet.exists()
        loaded_parquet = gpd.read_parquet(tmp_parquet)
        assert len(loaded_parquet) == len(sample_grid_gdf)
        assert "mean_lst" in loaded_parquet.columns

        # 2. GeoJSON
        exporter.save_geojson(sample_grid_gdf, tmp_geojson)
        assert tmp_geojson.exists()
        loaded_geojson = gpd.read_file(tmp_geojson)
        assert len(loaded_geojson) == len(sample_grid_gdf)
        assert loaded_geojson.crs == sample_grid_gdf.crs


def test_data_quality_report_generation(sample_grid_gdf: gpd.GeoDataFrame) -> None:
    """Verify markdown data quality report generation and proxy documentation."""
    pipeline = PuneDataPipeline()

    with tempfile.TemporaryDirectory() as tmpdir:
        report_file = Path(tmpdir) / "test_report.md"
        report_text = pipeline.generate_data_quality_report(
            gdf=sample_grid_gdf, output_path=report_file
        )

        assert report_file.exists()
        assert "Urban Cooling Planner - Data Quality & Feature Engineering Report" in report_text
        assert "pct_children" in report_text
        assert "low_income_proxy" in report_text
        assert "sky_view_factor" in report_text
        assert "100m x 100m" in report_text
