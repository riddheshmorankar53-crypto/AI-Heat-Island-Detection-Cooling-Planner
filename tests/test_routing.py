"""Unit tests for pedestrian heat-safe walking route engine."""

import numpy as np
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
    """Named landmarks snap to a nearby street node."""
    shaniwar = next(lm for lm in router.landmarks if lm["id"] == "SHANIWAR_WADA")
    node = router.find_nearest_node(shaniwar["lon"], shaniwar["lat"])
    assert node == shaniwar["node_id"]
    snapped = router.nodes[node]
    assert (
        np.hypot(
            (snapped["lon"] - shaniwar["lon"]) * 105000, (snapped["lat"] - shaniwar["lat"]) * 111000
        )
        < 200
    )


def test_route_geometry_is_continuous(router):
    """Street shapes are stitched in walking order: the line passes through each path node in turn."""
    res = router.route(73.8553, 18.5196, 73.8744, 18.5284, heat_avoidance_weight=2.0)
    for key in ("fastest_route", "heat_safe_route"):
        coords = [tuple(c) for c in res[key]["geometry"]["coordinates"]]
        pos = -1
        for nid in res[key]["properties"]["path_nodes"]:
            node_xy = (router.nodes[nid]["lon"], router.nodes[nid]["lat"])
            pos = coords.index(node_xy, pos + 1)  # raises if out of order or missing
        assert pos == len(coords) - 1
        steps = np.diff(np.array(coords), axis=0)
        drawn_m = np.hypot(steps[:, 0] * 105000, steps[:, 1] * 111000).sum()
        assert drawn_m == pytest.approx(res[key]["properties"]["distance_meters"], rel=0.05)


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
