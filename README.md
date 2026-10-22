# Urban Cooling Planner: AI + GIS + Graph ML System
**Pilot City: Pune, India (Central Core AOI)**

An end-to-end urban climate resilience system that detects urban heat island (UHI) hotspots, identifies vulnerable communities, recommends cooling interventions (trees, cool roofs, misting/water points, shaded transit shelters, and community cooling centers) under municipal budget constraints, and provides citizens with heat-safe pedestrian walking routes.

---

## 🏗️ System Architecture

```mermaid
flowchart LR
    A["Data Sources<br/>(Landsat-8, Sentinel-2,<br/>OSM, Census, DEM)"] --> B["Ingestion<br/>(STAC APIs,<br/>OSMnx, Rasterio)"]
    B --> C["Preprocessing<br/>(100m Metric Grid,<br/>NDVI, NDBI, Canopy, HVI)"]
    C --> D[("Processed Grid<br/>(GeoJSON, in-memory<br/>GeoPandas)")]
    D --> E["AI Engine"]
    
    subgraph E["AI Engine"]
        E1["Heat Prediction<br/>(LightGBM & GNN)"]
        E2["Vulnerability Assessment<br/>(HVI Index)"]
        E3["Intervention Recommender<br/>(OR-Tools MILP)"]
        E4["Cool Route Engine<br/>(Thermal Dijkstra)"]
    end
    
    E --> F["Application Layer"]
    
    subgraph F["Application Layer"]
        F1["Interactive Dashboard<br/>(React + Leaflet)"]
        F2["Scenario Simulator<br/>(Budget & Interventions)"]
        F3["Alerts & Early Warnings<br/>(Hotspots + Open-Meteo)"]
    end
    
    F --> G["Users<br/>(City Planners, Officers,<br/>Vulnerable Citizens)"]
```

---

## 🌟 Key Features

1. **100m Metric Grid Spatial Engine (`/data`)**:
   - High-resolution $100\text{m} \times 100\text{m}$ grid ($0.01\text{ km}^2$ per cell, 3,904 cells) over Central Pune (Shivajinagar, Shaniwar Wada, Budhwar Peth, Swargate, Mutha River corridor).
   - Features: Land Surface Temperature (LST), NDVI, NDBI, tree canopy %, impervious surface %, albedo, elevation, population density, vulnerable demographics ratio, and Heat Vulnerability Index (HVI).
   - **Real satellite data** (`python data/pipeline.py --live --force`): cloud-masked median composites from Microsoft Planetary Computer for Mar–May 2024 (pre-monsoon peak):
     - LST from the Landsat 8/9 Collection 2 Level-2 thermal band (6 scenes).
     - NDVI, NDBI, broadband albedo, tree canopy % and impervious % from Sentinel-2 L2A at 10m, averaged per 100m cell (6 scenes).
     - Elevation from the Copernicus GLO-30 DEM.
   - **Population** from Meta's High Resolution Settlement Layer (HRSL, 30m), aggregated to people per cell (~1.17 million in the AOI). Elderly (60+) and under-5 counts come from the same source.
   - **Distance to cooling facilities** from OpenStreetMap: 539 hospitals, clinics, community centres, libraries, shelters, drinking-water points, parks and gardens.
   - **Limitation**: HRSL's age layers apply one district-level census share to every pixel, so the vulnerable-population ratio is a uniform 17.5% across the AOI. It is real but has no variation within the city; ward-level census data would be needed for that. `is_synthetic: true` remains on the grid for this reason, and only for this column.
   - Offline mode (the default) generates a fully synthetic grid with the same schema.

2. **AI Heat Island & Vulnerability Prediction (`/ml`)**:
   - **Baseline**: Ridge Linear Regression ($R^2 = 0.686$).
   - **Gradient Boosted Tree (LightGBM)**: LST Regressor ($R^2 = 0.716$, $\text{RMSE} = 1.06^\circ\text{C}$) and Critical Hotspot Classifier ($F_1 = 0.663$, $\text{Recall} = 84.1\%$).
   - **Graph Neural Network (PyTorch Spatial GCN)**: 2-layer Spatial Graph Convolution over contiguous 100m cells. It is the best LST model ($R^2 = 0.760$, $\text{RMSE} = 0.98^\circ\text{C}$) because neighbouring cells carry real thermal context.
   - **Honest evaluation**: Fixed seed (`42`); train/val/test split by **500m spatial blocks**, so neighbouring (spatially autocorrelated) cells never straddle train and test; scaler fitted on train only; metrics saved to `ml/artifacts/metrics.json`.

3. **Google OR-Tools Cooling Intervention Optimizer (`/ml/optimizer`)**:
   - Solves a Mixed-Integer Linear Program (MILP) using SCIP/CBC.
   - Allocates 5 distinct cooling interventions (Urban Trees, Cool Roofs, Misting Points, Shaded Stops, and Cooling Refuges) within municipal budgets (e.g. ₹5 Lakh to ₹50 Lakh).
   - Enforces portfolio diversity, spatial suitability, per-cell limits, and 400m minimum separation between cooling refuges.
   - **Measured vs assumed cooling**: `ml/train.py` measures the tree effect on the real grid, comparing cells within the same 500m block and controlling for impervious cover and elevation: **0.029°C cooler per +1% canopy** (95% CI 0.024–0.034). With an assumed +10% canopy per ₹15,000 unit (`TREE_CANOPY_GAIN_PCT_PER_UNIT`, a tunable setting), trees give ~0.29°C per unit, about 5× less than the 1.5°C previously assumed. Cool roofs can't be measured from a single snapshot (Pune's brightest surfaces are hot bare soil), and misting, shade and refuges cool people rather than the land surface, so those four remain literature assumptions. The planner labels each value *measured* or *assumed*, and `GET /interventions/catalog` returns the basis.

4. **Heat-Safe Pedestrian Routing Engine (`/ml/routing`)**:
   - Real OpenStreetMap walking network (`--live`: 8,809 intersections, 11,710 street segments). Each street's shade and temperature are read from the real grid cells it passes through, sampled every 20m. Named landmarks snap to the nearest street node.
   - Dual-objective Dijkstra search calculating:
     - **Fastest Route**: Shortest physical distance in meters.
     - **Heat-Safe Route**: Minimizes direct solar thermal exposure and maximizes tree canopy shade, e.g. Shaniwar Wada → Pune Station: 12.9% less heat load for a 159m detour along the river corridor.

5. **Interactive Full-Stack GIS Dashboard (`/frontend` + `/backend`)**:
   - **Frontend**: React + Vite + Leaflet with dynamic choropleths (LST, NDVI, HVI, Hotspots), intervention influence radius buffers, route comparison lines, budget sliders, and a Model Transparency Modal.
   - **Backend**: FastAPI REST service serving the grid from an in-memory GeoPandas frame (no database needed; the data pipeline can optionally export to PostGIS).

6. **Heat Hazard Alerts (`GET /api/v1/alerts`)**:
   - Every critical hotspot ranked by Heat Vulnerability Index: top 10% `extreme`, next 40% `high`, the rest `moderate`.
   - **Early warning**: a 3-day Open-Meteo forecast for Pune (free, no key) raises every alert one level when any day reaches 40°C (IMD heatwave threshold) and two levels at 45°C. If the forecast is unreachable, alerts fall back to satellite data only.
   - Each alert carries concrete actions (misting hours, temporary shelter where no cooling facility is within a 5-minute walk (400m), door-to-door checks where a cell holds 100+ elderly residents and young children). Clicking an alert in the dashboard flies the map to that cell.

---

## 📊 Model Benchmarks (Real Satellite Data, Spatial Block Split, Test N=610, Seed 42)

> Generated by `ml/train.py` on held-out 500m spatial blocks of the real grid (Landsat / Sentinel-2 / DEM / HRSL / OSM). Earlier versions of this README reported $R^2 \approx 0.93$; those came from synthetic data whose temperature was a formula of the input features, scored with a random (leaky) split.

| Model Architecture | Task | R² Score / Acc | RMSE / F1 | Top Feature Drivers |
| :--- | :--- | :--- | :--- | :--- |
| **Ridge Regression** | LST Continuous | $R^2 = 0.686$ | $1.12^\circ\text{C}$ | (linear baseline) |
| **LightGBM Regressor** | LST Continuous | $R^2 = 0.716$ | $1.06^\circ\text{C}$ | Elevation (21%), Albedo (14%), NDBI (13%), Dist. to cooling (13%) |
| **Spatial GNN (PyTorch)** | LST with neighbour context | $R^2 = 0.760$ | $0.98^\circ\text{C}$ | Spatial neighbourhood adjacency |
| **LightGBM Classifier** | Hotspot Detection | $\text{Acc} = 91.2\%$ | $F_1 = 0.663$ | Recall 84.1% (53 of 63 test hotspots caught) |

Every model input is now observed data. The one exception is the vulnerable-population ratio, which is uniform (see above), so models gain nothing from it.

---

## 🚀 Quickstart Guide

### Prerequisites
- Python 3.11 or 3.13
- Node.js v18+ and npm
- (Optional) Docker & Docker Compose

### Step 1: Install Python Dependencies
```bash
# Full set: data pipeline, model training, tests and the API
pip install -e ".[dev]"
```
(`backend/requirements.txt` holds only the API's runtime dependencies, used by the Docker image.)

### Step 2: Generate Grid & Train Models (Reproducible Pipeline)
```bash
# 1. Generate 100m Pune grid and pedestrian network
#    --live --force pulls Landsat / Sentinel-2 / DEM / HRSL population and the OSM walking
#    network and facilities (~5 min, needs internet; Overpass calls are cached in ./cache).
#    Without flags it reuses data/processed/, or builds a synthetic grid offline.
python data/pipeline.py --live --force

# 2. Train baseline, LightGBM, and Spatial GNN models (saves to ml/artifacts/)
python ml/train.py
```

### Step 3: Start FastAPI Backend
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
- API Docs: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/api/v1/health`

### Step 4: Start React Frontend
In a separate terminal:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 🐳 Docker (Alternative)

One container builds the React dashboard and serves it together with the API:
```bash
docker compose up --build
```
- Dashboard: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

---

## 🌍 Deploy a Live Demo (Render, free)

`render.yaml` is a Render Blueprint for the same container:
1. Sign in at [render.com](https://render.com) with GitHub.
2. **New → Blueprint**, pick this repository, then **Apply**.
3. When the build finishes, open `https://<service-name>.onrender.com`.

Free instances sleep after ~15 minutes idle, so the first visit afterwards takes about a minute. The API idles at ~125 MB and peaks around 330 MB (512 MB limit); optimizer runs are serialised because each solve holds ~100 MB.

---

## 🧪 Automated Testing

Run the full pytest suite (35 unit & integration tests):
```bash
pytest tests/ -v
```

All 35 tests cover:
- 100m grid dimensions, valid geometries, and environmental ranges.
- Feature pipeline spatial-block train/val/test split (no neighbour leakage).
- Satellite raster-to-cell mapping.
- ML regression & classification models.
- Google OR-Tools budget & spatial constraints, and the measured tree-cooling calibration.
- Pedestrian heat-safe routing pathfinding.
- FastAPI REST endpoints, including heat alerts and forecast escalation.

---

## 📁 Repository Structure

```
├── config/
│   ├── config.yaml               # Master application settings & parameters
│   └── pune_aoi.yaml             # Pune bounding box, CRS & landmark hubs
├── data/
│   ├── ingestion/
│   │   ├── satellite.py          # Landsat / Sentinel-2 / DEM composites on the grid
│   │   └── real_context.py       # HRSL population, OSM cooling facilities & walk network
│   ├── raw/                      # Raw vector & raster storage
│   ├── processed/                # Processed 100m grid GeoJSON & street network
│   ├── synthetic/                # Deterministic synthetic data generator
│   └── pipeline.py               # Data ingestion & grid generation pipeline
├── ml/
│   ├── features/builder.py       # Spatial feature engineering & train-val-test splitting
│   ├── models/
│   │   ├── baseline.py           # Ridge regression & logistic baselines
│   │   ├── gradient_boost.py     # LightGBM Regressor & Classifier
│   │   └── spatial_gnn.py        # PyTorch 2-layer Spatial Graph Convolution
│   ├── optimizer/
│   │   └── intervention_solver.py# Google OR-Tools MILP cooling planner
│   ├── routing/
│   │   └── heat_safe_router.py   # NetworkX pedestrian thermal routing engine
│   ├── artifacts/                # Serialized model weights & metrics.json
│   └── train.py                  # End-to-end reproducible training pipeline
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI entrypoint & CORS
│   │   ├── config.py             # Pydantic Settings
│   │   ├── database.py           # In-memory GeoPandas grid, summary & alert queries
│   │   ├── forecast.py           # Open-Meteo 3-day forecast for heatwave escalation
│   │   ├── schemas/              # Pydantic request/response schemas
│   │   └── routers/              # Grid, Interventions, Routing, Metrics endpoints
│   └── requirements.txt          # API runtime dependencies (Docker image)
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx        # App bar with live stats & modal button
│   │   │   ├── MapView.jsx       # Leaflet interactive map with choropleths & buffers
│   │   │   ├── LayerControl.jsx  # Choropleth & overlay toggles
│   │   │   ├── PlannerPanel.jsx  # Budget slider & OR-Tools MILP optimizer UI
│   │   │   ├── RoutePanel.jsx    # Dual-route origin/dest picker & comparison cards
│   │   │   ├── AlertsPanel.jsx   # Heat hazard alerts ranked by vulnerability
│   │   │   └── MetricsModal.jsx  # Live model metrics & transparency modal
│   │   ├── services/api.js       # Backend REST API client
│   │   ├── App.jsx
│   │   └── index.css             # Glassmorphism dark UI styling
│   ├── package.json
│   └── vite.config.js
├── notebooks/
│   └── 01_pune_heat_island_eda.ipynb # EDA, correlation analysis & evaluation
├── docs/
│   ├── architecture.md           # Mermaid diagrams & system architecture
│   ├── mathematical_formulation.md# Equations for HVI, MILP solver & routing
│   └── api_reference.md          # Complete REST API reference
├── tests/                        # 35 unit and integration tests
├── Dockerfile                    # Single container: dashboard + API
├── render.yaml                   # Render Blueprint (free live demo)
├── docker-compose.yml            # Runs the Dockerfile locally
└── .env.example                  # Environment configuration template
```

---

## 📜 License
MIT License. Built for the Pune Urban Resilience & Climate Adaptation Initiative.
