# System Architecture: Urban Cooling Planner

The **Urban Cooling Planner** is a full-stack AI, GIS, and Graph ML system designed to detect urban heat island (UHI) hotspots, identify vulnerable communities, optimize cooling interventions within municipal budgets, and provide citizens with heat-safe pedestrian walking routes. Pilot city: Pune, India.

## High-Level Architecture Diagram

```mermaid
flowchart TD
    subgraph DataLayer["Data & GIS Ingestion Layer"]
        A[Pune Pilot AOI Config\nEPSG:4326 / EPSG:32643] --> B[100m Regular Grid Generator]
        B --> C[Spatial Feature Engine\nNDVI, NDBI, Canopy, Impervious, Elevation, Population]
        C --> D[(Processed GeoJSON, in-memory GeoPandas)]
    end

    subgraph MLLayer["Machine Learning & Optimization Layer"]
        D --> E[Spatial Feature Pipeline\nSplit Before Scaling]
        E --> F1[Baseline Ridge Regressor\nR²=0.686]
        E --> F2[LightGBM Regressor & Classifier\nR²=0.716, Recall=84%]
        E --> F3[PyTorch Spatial GNN\nAdjacency Convolution R²=0.760]
        F1 & F2 & F3 --> G[Saved Artifacts & metrics.json]

        D --> H[Google OR-Tools MILP Solver\nTrees, Roofs, Water, Shaded Stops, Cooling Centers]
        D --> I[NetworkX Pedestrian Router\nFastest vs Heat-Safe Dual Routing]
    end

    subgraph BackendLayer["FastAPI REST Services"]
        G & H & I & D --> J[FastAPI Application]
        J --> K1["GET /api/v1/grid\nGeoJSON 100m Cells"]
        J --> K2["GET /api/v1/hotspots/summary\nHeat Island Statistics"]
        J --> K3["POST /api/v1/interventions/optimize\nBudget Constrained MILP"]
        J --> K4["POST /api/v1/routing/heat-safe\nDual Route GeoJSON"]
        J --> K5["GET /api/v1/ml/metrics\nModel Transparency"]
    end

    subgraph FrontendLayer["React + Vite + Leaflet Frontend"]
        K1 & K2 & K3 & K4 & K5 --> L[Interactive GIS Dashboard]
        L --> M1[100m Choropleth: LST, NDVI, HVI, Hotspots]
        L --> M2[Cooling Interventions Placement & Buffers]
        L --> M3[Fastest vs Heat-Safe Walking Routes]
        L --> M4[Model Metrics & Transparency Modal]
    end
```

## Core Modules & Design Rationale

### 1. Data Ingestion & 100m Metric Grid (`/data`)
- **Spatial Resolution**: 100m $\times$ 100m cells ($0.01\text{ km}^2$), projected via UTM Zone 43N (EPSG:32643) for meter-accurate distance computation and reprojected to EPSG:4326 for Leaflet compatibility.
- **Physical Grounding**: Grounded in Pune's topography (Mutha River cooling corridor, Shivajinagar/Pune Station rail transit hubs, Budhwar/Nana Peth dense historical cores, and Vetal Tekdi/Fergusson green hills).
- **Real Satellite Data** (`--live`): `data/ingestion/satellite.py` reads only the AOI window of Planetary Computer COGs (Landsat 8/9 LST, Sentinel-2 L2A, Copernicus DEM), cloud-masks each scene, takes a per-pixel median over up to 6 scenes, warps to a 10m grid aligned to the fishnet, and averages each 10×10 block into its 100m cell.
- **Synthetic Fallback**: Offline, a deterministic generator provides the same schema. Demographics are always synthetic and the grid stays tagged `is_synthetic: true`; `satellite_source` says where the satellite fields came from.

### 2. Data Store (`/backend/app/database.py`)
- The processed grid GeoJSON is loaded once into an in-memory GeoPandas frame; no database server is needed. (The data pipeline can still export the grid to PostGIS.)

### 3. Machine Learning & Spatial Graph ML (`/ml`)
- **Strict Pipeline Ordering**: Featurization pipelines are fitted exclusively on training splits (70%) and applied to validation (15%) and test (15%) splits to eliminate data leakage.
- **Spatial Block Split**: Partitions are whole 500m blocks, so neighbouring (spatially autocorrelated) cells never straddle train and test.
- **Model Hierarchy**:
  - Baseline: Ridge Linear Regression.
  - Tree Ensemble: LightGBM regressor predicting Land Surface Temperature (LST) and LightGBM classifier detecting Critical Hotspots.
  - Graph Neural Network: 2-layer Spatial Graph Convolution (GCN) in PyTorch capturing microclimate thermal diffusion across contiguous cells.
- **Transparency**: All metrics saved to `ml/artifacts/metrics.json` are genuine test set numbers; zero fabricated values.

### 4. Cooling Intervention Optimization (`/ml/optimizer`)
- Solves a Mixed-Integer Linear Program (MILP) using Google OR-Tools.
- Allocates 5 distinct cooling interventions (Urban Trees, Cool Roofs, Misting Points, Shaded Stops, Cooling Centers) within municipal budget constraints while enforcing portfolio diversity and spatial separation.

### 5. Heat-Safe Walking Routing Engine (`/ml/routing`)
- Pedestrian street network modeled with NetworkX.
- Dual-objective Dijkstra search calculating:
  - Standard shortest walking distance.
  - Heat-safe route minimizing thermal exposure and maximizing tree canopy shade.
