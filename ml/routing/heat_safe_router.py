"""
Pedestrian heat-safe walking route planner for Central Pune.

Implements dual-objective routing using NetworkX:
1. Standard Shortest Route (minimizing physical distance in meters)
2. Heat-Safe Route (minimizing thermal exposure via tree canopy shade and cooler corridors)
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import numpy as np

MAX_SNAP_M = 1000.0  # farthest a requested point may lie from a street graph node


class HeatSafeRouter:
    """NetworkX graph routing engine balancing distance vs thermal microclimate stress."""

    def __init__(self, network_path: str = "data/processed/pune_pedestrian_network.json") -> None:
        """Load pedestrian network graph."""
        self.network_path = Path(network_path)
        if not self.network_path.is_absolute():
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.network_path = base_dir / network_path

        with open(self.network_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        self.nodes = self.raw_data["nodes"]
        self.edges = self.raw_data["edges"]
        self.graph = self._build_graph()

    def _build_graph(self) -> nx.Graph:
        """Construct undirected NetworkX graph with thermal attributes."""
        G = nx.Graph()
        for nid, data in self.nodes.items():
            G.add_node(nid, **data)

        for edge in self.edges:
            u = edge["source"]
            v = edge["target"]
            length = edge["length_meters"]
            shade = edge["shade_factor"]
            heat = edge["heat_risk"]
            lst = edge["ambient_lst"]

            G.add_edge(
                u,
                v,
                length=length,
                shade=shade,
                heat=heat,
                ambient_lst=lst,
                geometry=edge["geometry"],
            )
        return G

    def find_nearest_node(self, lon: float, lat: float) -> str:
        """Find the closest street graph node to given coordinates."""
        best_node = None
        best_dist = float("inf")
        for nid, data in self.nodes.items():
            dx = (lon - data["lon"]) * 105000.0
            dy = (lat - data["lat"]) * 111000.0
            dist = float(np.hypot(dx, dy))
            if dist < best_dist:
                best_dist = dist
                best_node = nid
        if best_dist > MAX_SNAP_M:
            raise ValueError(
                f"Point ({lon}, {lat}) is {best_dist:.0f} m from the street network (limit {MAX_SNAP_M:.0f} m)"
            )
        return best_node

    def _calculate_route_metrics(self, path: List[str]) -> Dict[str, Any]:
        """Compute aggregate length, shade %, and thermal exposure for a node sequence."""
        total_len = 0.0
        total_shade_len = 0.0
        total_heat_exposure = 0.0
        coordinates = []

        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            edge_data = self.graph[u][v]
            seg_len = edge_data["length"]
            s = edge_data["shade"]
            h = edge_data["heat"]

            total_len += seg_len
            total_shade_len += seg_len * s
            # Thermal exposure is proportional to distance walked under unshaded solar heat
            total_heat_exposure += seg_len * h * (1.0 - s * 0.75)

            if i == 0:
                coordinates.append([self.nodes[u]["lon"], self.nodes[u]["lat"]])
            coordinates.append([self.nodes[v]["lon"], self.nodes[v]["lat"]])

        shade_pct = round((total_shade_len / max(1.0, total_len)) * 100.0, 1)
        est_walk_mins = round((total_len / 1000.0) / 4.5 * 60.0, 1)  # 4.5 km/h

        return {
            "path_nodes": path,
            "distance_meters": round(total_len, 1),
            "walk_time_minutes": est_walk_mins,
            "shade_percentage": shade_pct,
            "thermal_exposure_score": round(total_heat_exposure, 1),
            "coordinates": coordinates,
        }

    def route(
        self,
        origin_lon: float,
        origin_lat: float,
        dest_lon: float,
        dest_lat: float,
        heat_avoidance_weight: float = 1.5,
    ) -> Dict[str, Any]:
        """
        Calculate dual routes: standard fastest vs heat-safe pedestrian route.

        Args:
            origin_lon: Origin longitude.
            origin_lat: Origin latitude.
            dest_lon: Destination longitude.
            dest_lat: Destination latitude.
            heat_avoidance_weight: Gamma penalty for walking through unshaded hot streets.

        Returns:
            Dictionary comparing fastest route vs heat-safe route with GeoJSON geometries.
        """
        start_node = self.find_nearest_node(origin_lon, origin_lat)
        end_node = self.find_nearest_node(dest_lon, dest_lat)

        if start_node == end_node:
            raise ValueError(
                f"Origin and destination both snap to {start_node}; pick points further apart"
            )

        # 1. Shortest path (distance only)
        fastest_path = nx.shortest_path(
            self.graph, source=start_node, target=end_node, weight="length"
        )
        fastest_metrics = self._calculate_route_metrics(fastest_path)

        # 2. Heat-safe path (thermal cost weighted)
        # Cost = length * (1.0 + gamma * heat * (1.0 - shade))
        def thermal_weight(u: str, v: str, d: Dict[str, Any]) -> float:
            seg_len = d["length"]
            s = d["shade"]
            h = d["heat"]
            return seg_len * (1.0 + heat_avoidance_weight * h * (1.0 - s))

        cool_path = nx.shortest_path(
            self.graph, source=start_node, target=end_node, weight=thermal_weight
        )
        cool_metrics = self._calculate_route_metrics(cool_path)

        # Comparative benefits
        exp_diff = (
            fastest_metrics["thermal_exposure_score"] - cool_metrics["thermal_exposure_score"]
        )
        heat_reduction_pct = round(
            max(0.0, (exp_diff / max(1.0, fastest_metrics["thermal_exposure_score"])) * 100.0), 1
        )
        dist_detour_m = round(
            max(0.0, cool_metrics["distance_meters"] - fastest_metrics["distance_meters"]), 1
        )

        return {
            "origin": {
                "node_id": start_node,
                "lon": self.nodes[start_node]["lon"],
                "lat": self.nodes[start_node]["lat"],
            },
            "destination": {
                "node_id": end_node,
                "lon": self.nodes[end_node]["lon"],
                "lat": self.nodes[end_node]["lat"],
            },
            "fastest_route": {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": fastest_metrics["coordinates"],
                },
                "properties": {
                    "route_type": "fastest",
                    "distance_meters": fastest_metrics["distance_meters"],
                    "walk_time_minutes": fastest_metrics["walk_time_minutes"],
                    "shade_percentage": fastest_metrics["shade_percentage"],
                    "thermal_exposure": fastest_metrics["thermal_exposure_score"],
                    "path_nodes": fastest_metrics["path_nodes"],
                },
            },
            "heat_safe_route": {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": cool_metrics["coordinates"],
                },
                "properties": {
                    "route_type": "heat_safe",
                    "distance_meters": cool_metrics["distance_meters"],
                    "walk_time_minutes": cool_metrics["walk_time_minutes"],
                    "shade_percentage": cool_metrics["shade_percentage"],
                    "thermal_exposure": cool_metrics["thermal_exposure_score"],
                    "path_nodes": cool_metrics["path_nodes"],
                },
            },
            "comparison": {
                "heat_stress_reduction_pct": heat_reduction_pct,
                "extra_distance_meters": dist_detour_m,
                "extra_walk_minutes": round(
                    max(
                        0.0,
                        cool_metrics["walk_time_minutes"] - fastest_metrics["walk_time_minutes"],
                    ),
                    1,
                ),
                "shade_increase_pct": round(
                    cool_metrics["shade_percentage"] - fastest_metrics["shade_percentage"], 1
                ),
            },
        }
