"""Unit tests for ML regression and classification models."""

import numpy as np
import pandas as pd
import pytest

from ml.models.baseline import BaselineHotspotClassifier, BaselineLSTModel
from ml.models.gradient_boost import GradientBoostHotspotClassifier, GradientBoostLSTModel


@pytest.fixture
def mock_data():
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame(
        {
            "ndbi": rng.uniform(-0.2, 0.6, size=n),
            "ndvi": rng.uniform(0.1, 0.6, size=n),
            "impervious_pct": rng.uniform(10, 90, size=n),
            "tree_canopy_pct": rng.uniform(5, 50, size=n),
            "elevation_m": rng.uniform(540, 620, size=n),
        }
    )
    y_reg = (
        34.0 + 0.1 * X["impervious_pct"] - 0.08 * X["tree_canopy_pct"] + rng.normal(0, 0.5, size=n)
    )
    y_cls = (y_reg > np.percentile(y_reg, 80)).astype(int)
    return X, y_reg, y_cls


def test_baseline_models(mock_data):
    X, y_reg, y_cls = mock_data
    baseline_reg = BaselineLSTModel(random_state=42)
    baseline_reg.fit(X.to_numpy(), y_reg.to_numpy())
    metrics_reg = baseline_reg.evaluate(X.to_numpy(), y_reg.to_numpy())

    assert "rmse" in metrics_reg
    assert "r2" in metrics_reg
    assert metrics_reg["r2"] > 0.5

    baseline_cls = BaselineHotspotClassifier(random_state=42)
    baseline_cls.fit(X.to_numpy(), y_cls.to_numpy())
    metrics_cls = baseline_cls.evaluate(X.to_numpy(), y_cls.to_numpy())
    assert "accuracy" in metrics_cls
    assert "f1_score" in metrics_cls


def test_gradient_boost_models(mock_data):
    X, y_reg, y_cls = mock_data
    lgbm_reg = GradientBoostLSTModel(n_estimators=30, random_state=42)
    lgbm_reg.fit(X, y_reg)
    metrics_reg = lgbm_reg.evaluate(X, y_reg)
    assert metrics_reg["r2"] > 0.6

    lgbm_cls = GradientBoostHotspotClassifier(n_estimators=30, random_state=42)
    lgbm_cls.fit(X, y_cls)
    metrics_cls = lgbm_cls.evaluate(X, y_cls)
    assert metrics_cls["accuracy"] > 0.7
