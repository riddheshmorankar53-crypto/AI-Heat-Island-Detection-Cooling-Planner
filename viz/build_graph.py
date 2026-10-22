"""
Build a feature-similarity graph over the Pune 100m grid and render it as an
Obsidian-style interactive network (viz/graph.html).

Nodes = grid cells. Each cell links to its K most similar cells in the
standardised 9-feature space (kNN graph), weighted by similarity.
Communities = Louvain; node size = degree.

Run from the project root:  python viz/build_graph.py
"""

import json
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
VIZ_DIR = BASE_DIR / "viz"
K = 5
D3_CDN_TAG = '<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>'
RESOLUTION = (
    0.4  # Louvain resolution: 0.4 gives 6 communities (fits the 7-colour palette); 1.0 gives 15
)

FEATURES = {
    "ndbi": "built-up index",
    "ndvi": "vegetation",
    "tree_canopy_pct": "tree canopy",
    "impervious_pct": "impervious surface",
    "albedo": "albedo",
    "elevation_m": "elevation",
    "pop_density_sqkm": "population density",
    "vuln_population_ratio": "vulnerable population",
    "dist_cooling_facility_m": "distance to cooling",
}


def build_knn_graph(gdf: gpd.GeoDataFrame, X: np.ndarray) -> nx.Graph:
    dist, idx = NearestNeighbors(n_neighbors=K + 1).fit(X).kneighbors(X)
    scale = np.median(dist[:, 1:])
    G = nx.Graph()
    G.add_nodes_from(range(len(gdf)))
    for i in range(len(gdf)):
        for d, j in zip(dist[i, 1:], idx[i, 1:]):  # column 0 is the cell itself
            G.add_edge(i, int(j), weight=round(float(np.exp(-d / scale)), 4))
    return G


def name_community(members: list, Z: np.ndarray, lst: np.ndarray) -> str:
    """Name a community by its mean LST and its most distinctive feature."""
    mean_z = Z[members].mean(axis=0)
    top = int(np.argmax(np.abs(mean_z)))
    direction = "high" if mean_z[top] > 0 else "low"
    return f"{lst[members].mean():.1f}°C · {direction} {list(FEATURES.values())[top]}"


def main() -> None:
    gdf = gpd.read_file(BASE_DIR / "data" / "processed" / "pune_100m_grid.geojson")
    Z = StandardScaler().fit_transform(gdf[list(FEATURES)].to_numpy())
    lst = gdf["lst_celsius"].to_numpy()

    G = build_knn_graph(gdf, Z)
    communities = sorted(
        nx.community.louvain_communities(G, weight="weight", resolution=RESOLUTION, seed=42),
        key=len,
        reverse=True,
    )
    comm_of = {n: c for c, members in enumerate(communities) for n in members}
    degree = dict(G.degree())

    data = {
        "communities": [
            {"id": c, "name": name_community(sorted(m), Z, lst), "size": len(m)}
            for c, m in enumerate(communities)
        ],
        "nodes": [
            {
                "id": int(i),
                "label": r.cell_id,
                "community": comm_of[i],
                "size": degree[i],
                "lst": round(float(r.lst_celsius), 1),
                "hvi": round(float(r.heat_vulnerability_index), 2),
                "hotspot": bool(r.is_critical_hotspot),
            }
            for i, r in gdf.iterrows()
        ],
        "edges": [{"source": u, "target": v, "weight": w} for u, v, w in G.edges(data="weight")],
    }

    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (VIZ_DIR / "graph.json").write_text(payload, encoding="utf-8")
    html = (
        (VIZ_DIR / "template.html")
        .read_text(encoding="utf-8")
        .replace("/*GRAPH_DATA*/null", payload)
    )
    d3_local = VIZ_DIR / "d3.v7.min.js"
    if d3_local.exists():  # inline D3 so the page works offline; otherwise it loads from the CDN
        html = html.replace(
            D3_CDN_TAG, "<script>" + d3_local.read_text(encoding="utf-8") + "</script>"
        )
    (VIZ_DIR / "graph.html").write_text(html, encoding="utf-8")
    print(
        f"D3: {'inlined from ' + d3_local.name if d3_local.exists() else 'loaded from CDN (no viz/d3.v7.min.js found)'}"
    )

    print(
        f"Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}  Communities: {len(communities)}"
    )
    print(
        f"Connected components: {nx.number_connected_components(G)}  Degree range: {min(degree.values())}-{max(degree.values())}"
    )
    print(f"Modularity: {nx.community.modularity(G, communities, weight='weight'):.3f}")
    for c in data["communities"]:
        print(f"  [{c['id']}] {c['size']:4d} cells  {c['name']}")
    print(f"Wrote {VIZ_DIR / 'graph.json'} and {VIZ_DIR / 'graph.html'}")


if __name__ == "__main__":
    main()
