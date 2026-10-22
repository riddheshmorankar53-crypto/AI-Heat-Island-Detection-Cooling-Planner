"""Unit tests for spatial feature engineering."""

import pytest

from data.pipeline import PuneDataPipeline
from ml.features.builder import FEATURE_COLUMNS, SpatialFeaturePipeline


@pytest.fixture(scope="module")
def grid_gdf():
    pipeline = PuneDataPipeline()
    return pipeline.generate_and_save_grid()


def test_train_val_test_split_integrity(grid_gdf):
    """Test that split occurs before scaling and partitions are disjoint."""
    pipeline = SpatialFeaturePipeline(seed=42)
    (
        X_train,
        X_val,
        X_test,
        y_reg_train,
        y_reg_val,
        y_reg_test,
        y_cls_train,
        y_cls_val,
        y_cls_test,
    ) = pipeline.prepare_splits(grid_gdf, test_size=0.15, val_size=0.15)

    # Check partition disjointness
    train_indices = set(X_train.index)
    val_indices = set(X_val.index)
    test_indices = set(X_test.index)

    assert train_indices.isdisjoint(val_indices)
    assert train_indices.isdisjoint(test_indices)
    assert val_indices.isdisjoint(test_indices)
    assert len(train_indices) + len(val_indices) + len(test_indices) == len(grid_gdf)

    # Check scaling behavior
    X_train_scaled = pipeline.fit_transform(X_train)
    X_test_scaled = pipeline.transform(X_test)

    assert X_train_scaled.shape == (len(X_train), len(FEATURE_COLUMNS))
    assert X_test_scaled.shape == (len(X_test), len(FEATURE_COLUMNS))
    # Mean of scaled training features should be approx 0
    assert abs(X_train_scaled.mean()) < 1e-2
