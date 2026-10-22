"""Unit tests for cooling intervention optimization engine."""

import pytest

from data.pipeline import PuneDataPipeline
from ml.optimizer.intervention_solver import CoolingInterventionOptimizer


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
