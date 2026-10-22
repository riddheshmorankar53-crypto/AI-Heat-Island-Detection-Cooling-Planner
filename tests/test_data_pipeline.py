"""
Unit and integration tests for the /data pipeline.

Validates:
- CRS consistency (EPSG:32643 metric internal, EPSG:4326 geographic output)
- Fishnet regular 100m x 100m grid cell geometry
- Presence and physical bounds of the grid schema that ml/ and backend/ consume
- GeoParquet and GeoJSON export round-trip integrity
- Data quality profiling report generation
"""

import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest

from data.pipeline import REPORT_SOURCES, PuneDataPipeline
from data.storage.exporter import DataExporter
from data.synthetic.generator import PuneSyntheticGridGenerator
from ml.features.builder import FEATURE_COLUMNS


@pytest.fixture(scope="module")
def sample_grid_gdf() -> gpd.GeoDataFrame:
    """Generate a lightweight synthetic grid for testing."""
    bbox = {"min_lat": 18.5000, "min_lon": 73.8400, "max_lat": 18.5200, "max_lon": 73.8600}
    gdf = PuneSyntheticGridGenerator(bbox=bbox, resolution_meters=100.0, seed=42).generate()
    gdf["satellite_source"] = "synthetic"
    return gdf


def test_crs_consistency(sample_grid_gdf: gpd.GeoDataFrame) -> None:
    """Verify final output is EPSG:4326 and internal coordinate validity."""
    assert sample_grid_gdf.crs is not None
    assert sample_grid_gdf.crs.to_string() == "EPSG:4326"
    assert sample_grid_gdf.geometry.is_valid.all()


def test_grid_alignment_and_area(sample_grid_gdf: gpd.GeoDataFrame) -> None:
    """Verify that re-projecting to metric UTM Zone 43N yields ~10,000 m² cells."""
    gdf_utm = sample_grid_gdf.to_crs("EPSG:32643")
    np.testing.assert_allclose(gdf_utm.geometry.area, 10000.0, rtol=0.01)


def test_feature_presence_and_bounds(sample_grid_gdf: gpd.GeoDataFrame) -> None:
    """Every column the models and API read exists and is physically plausible."""
    required = FEATURE_COLUMNS + [
        "cell_id",
        "centroid_lon",
        "centroid_lat",
        "lst_celsius",
        "lst_anomaly_celsius",
        "population",
        "heat_vulnerability_index",
        "is_critical_hotspot",
        "is_transit_hub",
    ]
    for col in required:
        assert col in sample_grid_gdf.columns, f"Missing feature column: {col}"

    g = sample_grid_gdf
    assert g["lst_celsius"].between(20.0, 55.0).all()
    assert g["ndvi"].between(-1.0, 1.0).all() and g["ndbi"].between(-1.0, 1.0).all()
    assert g["tree_canopy_pct"].between(0.0, 100.0).all()
    assert g["impervious_pct"].between(0.0, 100.0).all()
    assert g["elevation_m"].between(450.0, 900.0).all()
    assert (g["population"] >= 0).all() and (g["dist_cooling_facility_m"] >= 0).all()
    assert g["heat_vulnerability_index"].between(0.0, 1.0).all()
    assert g["is_critical_hotspot"].dtype == bool
    assert g["is_synthetic"].all()


def test_geoparquet_and_geojson_export(sample_grid_gdf: gpd.GeoDataFrame) -> None:
    """Verify GeoParquet and GeoJSON export and roundtrip loading."""
    exporter = DataExporter()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_parquet = Path(tmpdir) / "test_grid.parquet"
        tmp_geojson = Path(tmpdir) / "test_grid.geojson"

        exporter.save_geoparquet(sample_grid_gdf, tmp_parquet)
        loaded_parquet = gpd.read_parquet(tmp_parquet)
        assert len(loaded_parquet) == len(sample_grid_gdf)
        assert "lst_celsius" in loaded_parquet.columns

        exporter.save_geojson(sample_grid_gdf, tmp_geojson)
        loaded_geojson = gpd.read_file(tmp_geojson)
        assert len(loaded_geojson) == len(sample_grid_gdf)
        assert loaded_geojson.crs == sample_grid_gdf.crs


def test_data_quality_report_generation(sample_grid_gdf: gpd.GeoDataFrame) -> None:
    """Report profiles every grid column and labels offline data as synthetic."""
    pipeline = PuneDataPipeline()

    with tempfile.TemporaryDirectory() as tmpdir:
        report_file = Path(tmpdir) / "test_report.md"
        report_text = pipeline.generate_data_quality_report(
            gdf=sample_grid_gdf, output_path=report_file
        )

        assert report_file.exists()
        assert "Urban Cooling Planner - Data Quality & Feature Engineering Report" in report_text
        assert "100m x 100m" in report_text
        for col in REPORT_SOURCES:
            assert f"`{col}`" in report_text
        assert "| `lst_celsius` |" in report_text and "| Synthetic |" in report_text
        assert "Offline: deterministic synthetic generator" in report_text
