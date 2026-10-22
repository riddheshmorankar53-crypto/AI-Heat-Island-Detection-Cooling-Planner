"""Unit tests for cooling intervention optimization engine."""

import json

import numpy as np
import pandas as pd
import pytest

from data.pipeline import PuneDataPipeline
from ml.features.builder import spatial_block_ids
from ml.optimizer.calibration import estimate_canopy_cooling
from ml.optimizer.intervention_solver import (
    INTERVENTION_CATALOG,
    METRICS_PATH,
    TREE_CANOPY_GAIN_PCT_PER_UNIT,
    CoolingInterventionOptimizer,
)


def test_canopy_effect_survives_neighbourhood_confounding():
    """Recover a known canopy effect when hot blocks also have fewer trees."""
    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame(
        {
            "centroid_lon": 73.82 + rng.uniform(0, 0.06, n),
            "centroid_lat": 18.49 + rng.uniform(0, 0.05, n),
            "impervious_pct": rng.uniform(0, 100, n),
            "elevation_m": rng.uniform(540, 600, n),
        }
    )
    blocks = spatial_block_ids(df)
    heat_of = {b: rng.normal(0, 2) for b in np.unique(blocks)}
    block_heat = np.array([heat_of[b] for b in blocks])
    df["tree_canopy_pct"] = np.clip(40 - 8 * block_heat + rng.normal(0, 15, n), 0, 100)
    df["lst_celsius"] = (
        40
        - 0.05 * df["tree_canopy_pct"]
        + 0.02 * df["impervious_pct"]
        + block_heat
        + rng.normal(0, 0.5, n)
    )

    est = estimate_canopy_cooling(df, n_boot=50)
    naive = np.polyfit(df["tree_canopy_pct"], df["lst_celsius"], 1)[0]
    assert est["ci95"][0] <= -0.05 <= est["ci95"][1]
    assert naive < -0.08  # a pooled fit would have overstated the cooling


def test_tree_cooling_comes_from_measured_effect():
    """The tree catalog entry uses the effect ml/train.py measured, not a hard-coded value."""
    effect = json.loads(METRICS_PATH.read_text(encoding="utf-8"))["intervention_effects"][
        "tree_canopy"
    ]
    trees = INTERVENTION_CATALOG["trees"]
    expected = -effect["lst_change_per_canopy_pct"] * TREE_CANOPY_GAIN_PCT_PER_UNIT
    assert trees["delta_celsius"] == pytest.approx(expected, abs=0.01)
    assert trees["basis"].startswith("Measured in Pune")
    assert all(v["basis"] for v in INTERVENTION_CATALOG.values())


@pytest.fixture(scope="module")
def sample_grid():
    pipeline = PuneDataPipeline()
    return pipeline.generate_and_save_grid().head(200)


def test_optimizer_budget_constraint(sample_grid):
    """Test that optimizer adheres to the budget constraint."""
    opt = CoolingInterventionOptimizer()
    budget = 500000.0  # 5 Lakh INR
    res = opt.optimize(sample_grid, budget_inr=budget)

    assert res["status"] in ("OPTIMAL", "FEASIBLE")
    assert res["summary"]["total_cost_inr"] <= budget
    assert res["summary"]["interventions_count"] > 0
    assert len(res["features"]) == res["summary"]["interventions_count"]

    # Verify features GeoJSON validity
    for f in res["features"]:
        assert f["type"] == "Feature"
        assert f["geometry"]["type"] == "Point"
        assert len(f["geometry"]["coordinates"]) == 2
        assert "intervention_type" in f["properties"]
        assert "cost_inr" in f["properties"]


def test_optimizer_empty_types_error(sample_grid):
    """Test that invalid types raise ValueError."""
    opt = CoolingInterventionOptimizer()
    with pytest.raises(ValueError):
        opt.optimize(sample_grid, enabled_types=["invalid_type_xyz"])
