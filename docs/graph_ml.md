# Graph Machine Learning in the Urban Cooling Planner

The project builds **four separate graphs**, each used for a different graph technique.
No graph ML library (PyTorch Geometric, DGL) is used: the GCN is written by hand in PyTorch,
and NetworkX covers everything else.

| # | Graph | Nodes / edges | Technique | Code |
|---|---|---|---|---|
| 1 | Spatial grid graph | 3,904 cells / cells within ~150 m | Graph Convolutional Network (node regression) | `ml/models/spatial_gnn.py`, `ml/train.py` |
| 2 | Same grid graph | same | Graph analytics (homophily, assortativity, receptive field) | `ml/graph_demo.py` |
| 3 | Feature-similarity graph | 3,904 cells / 5 nearest neighbours in feature space | kNN graph + Louvain community detection | `viz/build_graph.py` |
| 4 | Pedestrian street graph | 22 landmarks / 32 streets | Heat-weighted Dijkstra routing | `ml/routing/heat_safe_router.py` |

---

## 1. Spatial Graph Convolutional Network

| Part | How it's done |
|---|---|
| **Nodes** | 3,904 grid cells, each 100 m × 100 m |
| **Node features** | 9 per cell: `ndbi`, `ndvi`, `tree_canopy_pct`, `impervious_pct`, `albedo`, `elevation_m`, `pop_density_sqkm`, `vuln_population_ratio`, `dist_cooling_facility_m`. Standardised with scikit-learn `StandardScaler`. |
| **Edges** | Two cells are linked if their centroids are within 0.0018° (~150 m). In practice this links each cell to its 8 surrounding cells: degree min/mean/max = 3 / 7.81 / 8 (edges of the area have fewer). |
| **Adjacency matrix** | Dense N×N matrix computed with NumPy, with self-loops added |
| **Normalisation** | Symmetric, Â = D^-½ (A + I) D^-½, as in the standard GCN (Kipf & Welling, 2017) |
| **Layer** | `GraphConvLayer`: H′ = Â · H · W (learned linear transform, then neighbour averaging) |
| **Model** | `SpatialMicroclimateGNN`: 9 → 32 hidden → ReLU → dropout → 1 output |
| **Message passing** | 2 layers, so each prediction draws on cells up to 2 hops away. This models heat spreading between neighbouring blocks. |
| **Task** | Predict land surface temperature (LST) per cell: node regression, trained on the whole graph at once |
| **Training setup** | The model sees the full graph; the loss uses only training nodes. Split 70/15/15 (2,732 / 586 / 586). Adam (lr 0.01, weight decay 1e-4), mean squared error loss, targets standardised. |
| **Output** | `ml/artifacts/spatial_gnn.pt` |

### Results (test set, from `ml/artifacts/metrics.json`)

| Model | RMSE (°C) | MAE (°C) | R² |
|---|---|---|---|
| Ridge regression (baseline) | 0.610 | 0.479 | 0.931 |
| LightGBM | 0.617 | 0.491 | 0.930 |
| **Spatial GCN** | **0.876** | **0.689** | **0.858** |

The GCN currently does worse than the simpler models. The likely reason is that the data is
synthetic and each cell's temperature can be predicted from its own features, so looking at
neighbours adds little.

### Known issue
`config/config.yaml` sets `gnn.dropout: 0.15` and `gnn.num_layers: 2`, but `ml/train.py` passes
neither to the model. Dropout is always the class default (0.1) and the model always has 2 layers.

---

## 2. Graph analytics on the grid graph

`ml/graph_demo.py` turns the GCN adjacency matrix into a NetworkX graph and reports:

- **Structure:** node and edge counts, degree distribution, density, connected components, average clustering coefficient
- **Homophily:** correlation of LST across edges. Strong homophily is why message passing should help.
- **Label assortativity:** whether hotspot cells link to other hotspot cells (`nx.attribute_assortativity_coefficient`)
- **Receptive field:** the 1-hop and 2-hop neighbourhood around the hottest cell, found by breadth-first search (`single_source_shortest_path_length`)
- **Predictions:** GCN predictions and test-set errors drawn on the graph

Output: `ml/artifacts/graph_demo.png`. Run with `python -m ml.graph_demo`.

---

## 3. Feature-similarity graph and community detection

`viz/build_graph.py` produces `viz/graph.html`.

| Part | How it's done |
|---|---|
| **Graph** | k-nearest-neighbour graph in standardised feature space (K = 5), built with scikit-learn `NearestNeighbors`. It links cells that are *similar*, not cells that are close on the map. |
| **Edge weights** | exp(−d / median d), so more similar cells get stronger links |
| **Communities** | Louvain modularity optimisation (`nx.community.louvain_communities`, resolution 0.4, seed 42), giving about 6 communities. Modularity is reported. |
| **Community labels** | Mean LST plus the feature with the largest average standardised score, e.g. "41.2°C · high impervious surface" |
| **Centrality** | Node size shows degree |
| **Rendering** | D3.js v7 force-directed layout (Obsidian-style) |

Run with `python viz/build_graph.py`.

---

## 4. Heat-safe pedestrian routing

| Part | How it's done |
|---|---|
| **Graph** | Undirected, weighted NetworkX graph: 22 landmark nodes, 32 street edges (synthetic; OSMnx is meant to supply the real network) |
| **Edge attributes** | `length`, `shade`, `heat`, `ambient_lst` |
| **Algorithm** | Dijkstra's shortest path (`nx.shortest_path`), run twice: on distance, then on a heat-aware weight |
| **Heat-aware cost** | length × (1 + γ · heat · (1 − shade)), where γ (`heat_avoidance_weight`, 0–5) sets how strongly the route avoids heat |
| **Snapping to the map** | Nearest node by equirectangular distance, rejected beyond 1 km (`MAX_SNAP_M`) |
| **API** | `POST /api/v1/routing/heat-safe` returns the fastest and heat-safe routes as GeoJSON, plus a comparison |

---

## Related, but not graph ML

- **Intervention optimizer** (`ml/optimizer/intervention_solver.py`): a mixed-integer program solved with Google OR-Tools. Each intervention covers a fixed radius around its cell, but this isn't modelled as a graph.
- **LightGBM and Ridge/Logistic regression**: the non-graph models the GCN is compared against.

## Libraries

| Library | Role |
|---|---|
| PyTorch | GCN layers, training, saving the model |
| NetworkX | Graph structure, analytics, Louvain, Dijkstra |
| scikit-learn | `StandardScaler`, `NearestNeighbors`, baseline models |
| NumPy | Adjacency matrix and normalisation |
| GeoPandas / Shapely | Grid geometry and cell centroids |
| Matplotlib | `graph_demo.png` |
| D3.js v7 | Interactive similarity-graph page |
| FastAPI | Routing endpoint |
