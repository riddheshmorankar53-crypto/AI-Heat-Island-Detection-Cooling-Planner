"""Unit tests for pedestrian heat-safe walking route engine."""

import pytest

from ml.routing.heat_safe_router import HeatSafeRouter


@pytest.fixture(scope="module")
def router():
    return HeatSafeRouter()


def test_router_graph_integrity(router):
    """Test graph structure and attributes."""
    assert len(router.nodes) > 10
    assert len(router.edges) > 10
    assert router.graph.number_of_nodes() > 10
    assert router.graph.number_of_edges() > 10


def test_nearest_node_lookup(router):
    """Test finding nearest node to coordinates."""
    # Shaniwar Wada vicinity
    node = router.find_nearest_node(73.8553, 18.5196)
    assert node == "SHANIWAR_WADA"


def test_dual_route_calculation(router):
    """Test route calculation produces valid GeoJSON LineStrings and comparative metrics."""
    res = router.route(
        origin_lon=73.8553,
        origin_lat=18.5196,
        dest_lon=73.8744,
        dest_lat=18.5284,
        heat_avoidance_weight=2.0,
    )

    assert "fastest_route" in res
    assert "heat_safe_route" in res
    assert "comparison" in res

    fastest = res["fastest_route"]
    cool = res["heat_safe_route"]

    assert fastest["geometry"]["type"] == "LineString"
    assert cool["geometry"]["type"] == "LineString"
    assert len(fastest["geometry"]["coordinates"]) >= 2
    assert len(cool["geometry"]["coordinates"]) >= 2

    assert "shade_percentage" in fastest["properties"]
    assert "distance_meters" in fastest["properties"]
    assert "heat_stress_reduction_pct" in res["comparison"]


def test_rejects_point_outside_network(router):
    with pytest.raises(ValueError):
        router.route(origin_lon=0.0, origin_lat=0.0, dest_lon=73.8744, dest_lat=18.5284)


def test_rejects_same_origin_and_destination(router):
    with pytest.raises(ValueError):
        router.route(origin_lon=73.8553, origin_lat=18.5196, dest_lon=73.8553, dest_lat=18.5196)
