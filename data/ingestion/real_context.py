"""
Real population, cooling-facility and street-network context for the 100m grid
(used by `python data/pipeline.py --live`).

- Population, elderly (60+) and under-5 counts: Meta High Resolution Settlement Layer
  (HRSL, 30m, public cloud-optimised GeoTIFFs on AWS).
- Cooling facilities: OpenStreetMap hospitals, clinics, community centres, libraries,
  shelters, drinking water points, parks and gardens.
- Walking network: OpenStreetMap via OSMnx; each street takes canopy shade and LST from
  the grid cells it crosses.
"""

import time
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import numpy as np
import osmnx as ox
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import transform as shapely_transform

from data.ingestion.satellite import CELL_M, GRID_CRS, _read_to_grid, cell_values, grid_layout

HRSL_URL = "/vsicurl/https://dataforgood-fb-data.s3.amazonaws.com/hrsl-cogs/{0}/{0}-latest.vrt"
COOLING_TAGS = {
    "amenity": [
        "hospital",
        "clinic",
        "community_centre",
        "library",
        "shelter",
        "drinking_water",
    ],
    "leisure": ["park", "garden"],
}
EDGE_SAMPLE_M = 20.0  # spacing of points sampled along each street to read grid cells
OVERPASS_RETRIES = 3


def _overpass(fn, *args, **kwargs):
    """Call an OSMnx query, retrying when the public Overpass server is busy."""
    for attempt in range(1, OVERPASS_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == OVERPASS_RETRIES:
                raise
            print(f"Overpass query failed ({type(e).__name__}); retry {attempt} in 30s")
            time.sleep(30)


def _hrsl_people_per_cell(layer: str, layout: Dict[str, Any]) -> np.ndarray:
    """People per 100m cell from an HRSL count raster (30m, EPSG:4326)."""
    import rasterio

    href = HRSL_URL.format(layer)
    with rasterio.open(href) as src:
        res_deg = src.res[0]
    counts = np.nan_to_num(_read_to_grid(href, layout, nearest=True))  # NaN = uninhabited
    lat = 18.52  # pixel area varies <0.1% across the AOI
    pixel_m2 = (res_deg * 111_320 * np.cos(np.radians(lat))) * (res_deg * 110_574)
    return cell_values(counts / pixel_m2, layout) * CELL_M * CELL_M


def apply_real_population(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Replace synthetic population and vulnerable-population ratio with HRSL counts."""
    layout = grid_layout(gdf)
    people = _hrsl_people_per_cell("hrsl_general", layout)
    elderly = _hrsl_people_per_cell("hrsl_elderly_60_plus", layout)
    children = _hrsl_people_per_cell("hrsl_children_under_five", layout)
    print(f"HRSL population in AOI: {people.sum():,.0f}")

    out = gdf.copy()
    out["population"] = np.rint(people).astype(int)
    out["pop_density_sqkm"] = np.rint(people * 100.0).astype(int)  # 100m cell = 0.01 km²
    with np.errstate(all="ignore"):
        ratio = (elderly + children) / people
    ratio = np.where(people >= 1.0, ratio, np.nan)  # empty cells: no meaningful ratio
    out["vuln_population_ratio"] = np.round(np.nan_to_num(ratio, nan=np.nanmedian(ratio)), 4)
    return out


def apply_real_cooling_distance(gdf: gpd.GeoDataFrame, bbox: List[float]) -> gpd.GeoDataFrame:
    """Distance from each cell centroid to the nearest OSM cooling facility (m)."""
    facilities = _overpass(ox.features_from_bbox, tuple(bbox), COOLING_TAGS).to_crs(GRID_CRS)
    print(f"OSM cooling facilities: {len(facilities)}")
    centroids = gpd.GeoDataFrame(geometry=gdf.to_crs(GRID_CRS).geometry.centroid, crs=GRID_CRS)
    nearest = gpd.sjoin_nearest(
        centroids, facilities[["geometry"]], how="left", distance_col="dist"
    )
    dist = nearest.groupby(level=0)["dist"].min().reindex(centroids.index)

    out = gdf.copy()
    out["dist_cooling_facility_m"] = np.round(dist.to_numpy(), 1)
    return out


def _cell_lookup(grid: gpd.GeoDataFrame) -> Tuple[Dict[str, Any], np.ndarray]:
    """Layout plus a (rows, cols) array mapping each 100m block to its grid row index."""
    layout = grid_layout(grid)
    k = layout["k"]
    h, w = layout["shape"][0] // k, layout["shape"][1] // k
    index = np.full((h, w), -1, dtype=int)
    index[layout["rows"], layout["cols"]] = np.arange(len(grid))
    return layout, index


def build_walk_network(
    grid: gpd.GeoDataFrame,
    bbox: List[float],
    landmarks: List[Tuple[str, float, float, str]],
) -> Dict[str, Any]:
    """Real OSM walking network in the router's JSON format, with grid-derived heat attributes."""
    G = ox.convert.to_undirected(_overpass(ox.graph_from_bbox, tuple(bbox), network_type="walk"))
    to_utm = Transformer.from_crs("EPSG:4326", GRID_CRS, always_xy=True).transform
    print(f"OSM walk network: {len(G.nodes)} nodes, {len(G.edges)} edges")

    layout, index = _cell_lookup(grid)
    x0, y1 = layout["transform"].c, layout["transform"].f
    canopy = grid["tree_canopy_pct"].to_numpy() / 100.0
    lst = grid["lst_celsius"].to_numpy()
    lst_lo, lst_hi = float(lst.min()), float(lst.max())

    def sample_cells(line_utm: LineString) -> np.ndarray:
        n = max(2, int(line_utm.length // EDGE_SAMPLE_M) + 1)
        pts = [line_utm.interpolate(d) for d in np.linspace(0, line_utm.length, n)]
        cols = ((np.array([p.x for p in pts]) - x0) // CELL_M).astype(int)
        rows = ((y1 - np.array([p.y for p in pts])) // CELL_M).astype(int)
        ok = (rows >= 0) & (rows < index.shape[0]) & (cols >= 0) & (cols < index.shape[1])
        cells = index[rows[ok], cols[ok]]
        return cells[cells >= 0]

    nodes = {
        str(n): {"id": str(n), "lon": round(d["x"], 6), "lat": round(d["y"], 6), "type": "osm"}
        for n, d in G.nodes(data=True)
    }

    edges: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for u, v, d in G.edges(data=True):
        key = (min(u, v), max(u, v))
        if key in edges and edges[key]["length_meters"] <= d["length"]:
            continue  # keep the shortest of parallel streets between the same nodes
        geom = d.get("geometry") or LineString(
            [(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]
        )
        cells = sample_cells(shapely_transform(to_utm, geom))
        if len(cells) == 0:
            continue  # street lies outside the grid
        seg_lst = float(lst[cells].mean())
        highway = d.get("highway", "footway")
        edges[key] = {
            "source": str(u),
            "target": str(v),
            "length_meters": round(float(d["length"]), 1),
            "shade_factor": round(float(canopy[cells].mean()), 3),
            "ambient_lst": round(seg_lst, 1),
            "heat_risk": round(max(0.05, (seg_lst - lst_lo) / (lst_hi - lst_lo)), 3),
            "road_type": highway[0] if isinstance(highway, list) else str(highway),
            "geometry": [[round(x, 6), round(y, 6)] for x, y in geom.coords],
        }
    edges = list(edges.values())

    # Named landmarks snap to the nearest real street node
    node_ids = list(nodes)
    node_xy = np.array([[nodes[n]["lon"], nodes[n]["lat"]] for n in node_ids])
    cell_index = grid.sindex
    lm_out = []
    for lid, lon, lat, ltype in landmarks:
        dx = (node_xy[:, 0] - lon) * 105_000.0
        dy = (node_xy[:, 1] - lat) * 111_000.0
        hits = list(cell_index.query(Point(lon, lat)))
        lm_out.append(
            {
                "id": lid,
                "lon": lon,
                "lat": lat,
                "type": ltype,
                "node_id": node_ids[int(np.argmin(np.hypot(dx, dy)))],
                "ambient_lst": round(float(grid.iloc[hits[0]]["lst_celsius"]), 1) if hits else None,
            }
        )

    return {
        "metadata": {
            "city": "Pune",
            "aoi": "Central Core",
            "source": "OpenStreetMap via OSMnx (walk network)",
            "nodes_count": len(nodes),
            "edges_count": len(edges),
            "is_synthetic": False,
        },
        "nodes": nodes,
        "edges": edges,
        "landmarks": lm_out,
    }
