"""
Graph ML demo: the Pune 100m grid as a graph.

Nodes = 100m grid cells (9 features each), edges = cells within ~150m
(same rule the Spatial GCN trains on). Prints graph statistics, runs the
trained GCN from ml/artifacts/spatial_gnn.pt, and saves a 4-panel figure.

Run from the project root:  python -m ml.graph_demo
"""

from pathlib import Path

import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from ml.features.builder import FEATURE_COLUMNS, SpatialFeaturePipeline
from ml.models.spatial_gnn import SpatialMicroclimateGNN, build_spatial_adjacency

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS = BASE_DIR / "ml" / "artifacts"


def build_graph(gdf: gpd.GeoDataFrame, adj_norm: torch.Tensor) -> nx.Graph:
    """Turn the GCN adjacency matrix into a NetworkX graph with cell attributes."""
    rows, cols = np.nonzero(np.triu(adj_norm.numpy(), k=1))
    G = nx.Graph()
    for i, r in gdf.iterrows():
        G.add_node(
            i,
            pos=(r.centroid_lon, r.centroid_lat),
            lst=r.lst_celsius,
            hotspot=int(r.is_critical_hotspot),
        )
    G.add_edges_from(zip(rows.tolist(), cols.tolist()))
    return G


def print_stats(G: nx.Graph) -> None:
    degrees = [d for _, d in G.degree()]
    lst = np.array([G.nodes[n]["lst"] for n in G])
    u, v = np.array(G.edges()).T
    print("=" * 60)
    print("PUNE HEAT GRID AS A GRAPH")
    print("=" * 60)
    print(f"Nodes (100m cells):          {G.number_of_nodes()}")
    print(f"Edges (cells within ~150m):  {G.number_of_edges()}")
    print(f"Node features:               {len(FEATURE_COLUMNS)} -> {', '.join(FEATURE_COLUMNS)}")
    print(f"Degree min / mean / max:     {min(degrees)} / {np.mean(degrees):.2f} / {max(degrees)}")
    print(f"Density:                     {nx.density(G):.5f}")
    print(f"Connected components:        {nx.number_connected_components(G)}")
    print(f"Avg clustering coefficient:  {nx.average_clustering(G):.3f}")
    # Homophily: do neighbours share temperature / hotspot label? This is why message passing helps.
    print(f"LST correlation across edges (homophily):  {np.corrcoef(lst[u], lst[v])[0, 1]:.3f}")
    print(
        f"Hotspot label assortativity:               {nx.attribute_assortativity_coefficient(G, 'hotspot'):.3f}"
    )


def run_gnn(gdf: gpd.GeoDataFrame, adj_norm: torch.Tensor):
    """Load the trained GCN and predict LST for every node. Returns (preds, test_mask)."""
    # Re-create the exact seeded split from ml/train.py to recover target scaling and the test mask
    # (train.py does not persist target mean/std).
    pipe = SpatialFeaturePipeline(seed=42)
    X_train, _, X_test, y_train, *_ = pipe.prepare_splits(gdf, test_size=0.15, val_size=0.15)
    mean, std = float(y_train.mean()), float(y_train.std())

    scaler = joblib.load(ARTIFACTS / "scaler.joblib")
    x = torch.tensor(scaler.transform(gdf[FEATURE_COLUMNS]), dtype=torch.float32)

    state = torch.load(ARTIFACTS / "spatial_gnn.pt", map_location="cpu")
    model = SpatialMicroclimateGNN(
        in_features=len(FEATURE_COLUMNS), hidden_dim=state["conv1.linear.weight"].shape[0]
    )
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        preds = model(x, adj_norm).squeeze().numpy() * std + mean

    test_mask = np.zeros(len(gdf), dtype=bool)
    test_mask[X_test.index.to_numpy()] = True
    y = gdf["lst_celsius"].to_numpy()
    rmse = np.sqrt(np.mean((preds[test_mask] - y[test_mask]) ** 2))
    print(f"\nSpatial GCN test RMSE: {rmse:.3f} degC on {test_mask.sum()} held-out nodes")
    return preds, test_mask


def plot(
    G: nx.Graph, gdf: gpd.GeoDataFrame, preds: np.ndarray, test_mask: np.ndarray, out: Path
) -> None:
    pos = nx.get_node_attributes(G, "pos")
    lst = gdf["lst_celsius"].to_numpy()
    vmin, vmax = lst.min(), lst.max()
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # (a) Whole graph, nodes coloured by observed LST
    ax = axes[0, 0]
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.2, edge_color="#bbbbbb")
    sc = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=4, node_color=lst, cmap="inferno", vmin=vmin, vmax=vmax
    )
    fig.colorbar(sc, ax=ax, label="LST (°C)")
    ax.set_title(
        f"(a) Grid graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges — observed LST"
    )

    # (b) 2-hop receptive field of the 2-layer GCN around the hottest cell
    ax = axes[0, 1]
    center = int(np.argmax(lst))
    hops = nx.single_source_shortest_path_length(G, center, cutoff=4)
    sub = G.subgraph(hops)
    colors = [
        "red"
        if hops[n] == 0
        else "orange"
        if hops[n] == 1
        else "gold"
        if hops[n] == 2
        else "lightgrey"
        for n in sub
    ]
    nx.draw_networkx_edges(sub, pos, ax=ax, width=0.8, edge_color="#888888")
    nx.draw_networkx_nodes(
        sub, pos, ax=ax, node_size=90, node_color=colors, edgecolors="black", linewidths=0.4
    )
    for label, c in [
        ("target node", "red"),
        ("1-hop (layer 1)", "orange"),
        ("2-hop (layer 2)", "gold"),
        ("outside receptive field", "lightgrey"),
    ]:
        ax.scatter([], [], c=c, edgecolors="black", label=label)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4)
    ax.set_title(
        f"(b) Message passing: 2-layer GCN receptive field of hottest cell ({lst[center]:.1f} °C)"
    )

    # (c) GCN predictions on the graph
    ax = axes[1, 0]
    sc = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=4, node_color=preds, cmap="inferno", vmin=vmin, vmax=vmax
    )
    fig.colorbar(sc, ax=ax, label="Predicted LST (°C)")
    ax.set_title("(c) Spatial GCN predicted LST (all nodes)")

    # (d) Absolute error on held-out test nodes
    ax = axes[1, 1]
    err = np.abs(preds - lst)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=2, node_color="#dddddd")
    test_nodes = np.flatnonzero(test_mask).tolist()
    sc = nx.draw_networkx_nodes(
        G, pos, nodelist=test_nodes, ax=ax, node_size=10, node_color=err[test_mask], cmap="viridis"
    )
    fig.colorbar(sc, ax=ax, label="|error| (°C)")
    ax.set_title(f"(d) Test-node error — RMSE {np.sqrt(np.mean(err[test_mask] ** 2)):.3f} °C")

    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.set_axis_off()
    fig.suptitle("Urban Heat Island as a Graph — Pune 100m grid + Spatial GCN", fontsize=16)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved figure to {out}")


def main() -> None:
    gdf = gpd.read_file(BASE_DIR / "data" / "processed" / "pune_100m_grid.geojson")
    adj_norm = build_spatial_adjacency(gdf)
    G = build_graph(gdf, adj_norm)
    print_stats(G)
    preds, test_mask = run_gnn(gdf, adj_norm)
    plot(G, gdf, preds, test_mask, ARTIFACTS / "graph_demo.png")


if __name__ == "__main__":
    main()
