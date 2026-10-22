"""
Synthetic Data Generator for Pune Urban Cooling Planner.

Generates physically grounded 100m grid cells over Central Pune AOI with
environmental, satellite proxy, and socioeconomic vulnerability indicators.
All output features are clearly stamped with `is_synthetic: true`.
"""

from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon


class PuneSyntheticGridGenerator:
    """
    Generates a 100m resolution spatial grid with synthetic environmental and
    socioeconomic features grounded in Pune's topography and urban layout.
    """

    def __init__(
        self,
        bbox: Dict[str, float],
        resolution_meters: float = 100.0,
        crs_geo: str = "EPSG:4326",
        crs_proj: str = "EPSG:32643",
        seed: int = 42,
    ) -> None:
        """
        Initialize the synthetic grid generator.

        Args:
            bbox: Bounding box dict containing min_lat, min_lon, max_lat, max_lon.
            resolution_meters: Grid cell width and height in meters (default 100m).
            crs_geo: Source geographic CRS.
            crs_proj: Projected metric CRS for Pune (UTM 43N).
            seed: Random seed for reproducibility.
        """
        self.bbox = bbox
        self.resolution = resolution_meters
        self.crs_geo = crs_geo
        self.crs_proj = crs_proj
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Coordinate transformers
        self.to_proj = Transformer.from_crs(crs_geo, crs_proj, always_xy=True)
        self.to_geo = Transformer.from_crs(crs_proj, crs_geo, always_xy=True)

        # Key spatial anchor points in Pune (lon, lat)
        self.mutha_river_nodes: List[Tuple[float, float]] = [
            (73.8320, 18.4980),
            (73.8420, 18.5080),
            (73.8490, 18.5160),
            (73.8560, 18.5280),
            (73.8700, 18.5410),
        ]
        self.commercial_core = (73.8560, 18.5180)  # Budhwar Peth / Shaniwar Wada
        self.green_hill = (73.8310, 18.5280)  # Fergusson / Vetal Tekdi edge
        self.swargate_transit = (73.8580, 18.5018)  # Swargate
        self.pune_station = (73.8744, 18.5284)  # Pune Railway Station

    def _generate_bounding_grid(self) -> gpd.GeoDataFrame:
        """Create regular polygon grid in projected metric CRS."""
        min_x, min_y = self.to_proj.transform(self.bbox["min_lon"], self.bbox["min_lat"])
        max_x, max_y = self.to_proj.transform(self.bbox["max_lon"], self.bbox["max_lat"])

        x_coords = np.arange(min_x, max_x, self.resolution)
        y_coords = np.arange(min_y, max_y, self.resolution)

        cells = []
        cell_ids = []
        centroids_x = []
        centroids_y = []

        counter = 0
        for x in x_coords:
            for y in y_coords:
                poly = Polygon(
                    [
                        (x, y),
                        (x + self.resolution, y),
                        (x + self.resolution, y + self.resolution),
                        (x, y + self.resolution),
                    ]
                )
                cells.append(poly)
                cell_ids.append(f"PUNE_100M_{counter:05d}")
                centroids_x.append(x + self.resolution / 2.0)
                centroids_y.append(y + self.resolution / 2.0)
                counter += 1

        gdf = gpd.GeoDataFrame(
            {
                "cell_id": cell_ids,
                "x_proj": centroids_x,
                "y_proj": centroids_y,
            },
            geometry=cells,
            crs=self.crs_proj,
        )
        return gdf

    def _distance_to_point(
        self, x: np.ndarray, y: np.ndarray, pt: Tuple[float, float]
    ) -> np.ndarray:
        """Compute metric distance to a geographic coordinate point."""
        pt_x, pt_y = self.to_proj.transform(pt[0], pt[1])
        return np.hypot(x - pt_x, y - pt_y)

    def _distance_to_river_corridor(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute minimum distance to the Mutha river line segments."""
        min_dist = np.full_like(x, fill_value=1e9, dtype=np.float64)
        for i in range(len(self.mutha_river_nodes) - 1):
            p1_lon, p1_lat = self.mutha_river_nodes[i]
            p2_lon, p2_lat = self.mutha_river_nodes[i + 1]
            x1, y1 = self.to_proj.transform(p1_lon, p1_lat)
            x2, y2 = self.to_proj.transform(p2_lon, p2_lat)

            # Segment vector
            dx = x2 - x1
            dy = y2 - y1
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq == 0:
                continue

            # Projection factor t
            t = ((x - x1) * dx + (y - y1) * dy) / seg_len_sq
            t = np.clip(t, 0.0, 1.0)
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist = np.hypot(x - proj_x, y - proj_y)
            min_dist = np.minimum(min_dist, dist)

        return min_dist

    def generate(self) -> gpd.GeoDataFrame:
        """
        Generate complete 100m grid with synthetic features.

        Returns:
            GeoPandas GeoDataFrame in EPSG:4326 with full attributes.
        """
        gdf_proj = self._generate_bounding_grid()
        x = gdf_proj["x_proj"].to_numpy()
        y = gdf_proj["y_proj"].to_numpy()
        n = len(gdf_proj)

        # Reproject centroid coordinates back to geographic (lon, lat)
        lon_arr, lat_arr = self.to_geo.transform(x, y)
        gdf_proj["centroid_lon"] = np.round(lon_arr, 6)
        gdf_proj["centroid_lat"] = np.round(lat_arr, 6)

        # Spatial distances to anchor features
        dist_river = self._distance_to_river_corridor(x, y)
        dist_core = self._distance_to_point(x, y, self.commercial_core)
        dist_hill = self._distance_to_point(x, y, self.green_hill)
        dist_station = self._distance_to_point(x, y, self.pune_station)
        dist_swargate = self._distance_to_point(x, y, self.swargate_transit)

        # 1. Built-Up Index (NDBI: -0.2 to 0.7)
        # Higher near commercial core and transit hubs; lower near river and hills
        core_factor = np.exp(-dist_core / 1800.0)
        station_factor = np.exp(-dist_station / 1200.0)
        swargate_factor = np.exp(-dist_swargate / 1000.0)
        hill_damping = np.exp(-dist_hill / 1500.0)

        ndbi_base = (
            0.25
            + 0.35 * core_factor
            + 0.15 * station_factor
            + 0.15 * swargate_factor
            - 0.25 * hill_damping
        )
        ndbi_noise = self.rng.normal(0.0, 0.04, size=n)
        ndbi = np.clip(ndbi_base + ndbi_noise, -0.20, 0.70)

        # 2. Vegetation Index (NDVI: 0.05 to 0.65)
        # Anti-correlated with NDBI; enhanced near river corridor and hill
        river_cooling = np.exp(-dist_river / 600.0)
        ndvi_base = 0.12 + 0.38 * hill_damping + 0.22 * river_cooling - 0.15 * core_factor
        ndvi_noise = self.rng.normal(0.0, 0.03, size=n)
        ndvi = np.clip(ndvi_base + ndvi_noise, 0.05, 0.65)

        # 3. Tree Canopy Coverage (%)
        canopy_pct = np.clip(
            (ndvi - 0.10) * 1.35 * 100.0 + self.rng.normal(0, 3, size=n), 2.0, 75.0
        )

        # 4. Impervious Surface Ratio (%)
        impervious_pct = np.clip((ndbi + 0.20) * 95.0 + self.rng.normal(0, 4, size=n), 10.0, 95.0)

        # 5. Surface Albedo (0.08 to 0.32)
        # Dark asphalt / unpainted concrete: 0.10 - 0.15; reflective surfaces: 0.25+
        albedo = np.clip(
            0.14
            - 0.04 * core_factor
            + 0.06 * (1.0 - impervious_pct / 100.0)
            + self.rng.normal(0, 0.015, size=n),
            0.08,
            0.32,
        )

        # 6. Elevation (meters above sea level for Pune: ~530m to 640m)
        elevation_m = (
            545.0 + 80.0 * hill_damping - 15.0 * river_cooling + self.rng.normal(0, 2.5, size=n)
        )

        # 7. Land Surface Temperature (LST in Celsius: summer daytime peak)
        # Rural / hill baseline in Pune summer: ~33°C. Dense asphalt core: 42°C - 45°C.
        lst_base = (
            34.0
            + 11.0 * (impervious_pct / 100.0)
            - 8.5 * (canopy_pct / 100.0)
            - 2.8 * river_cooling
            - 0.03 * (elevation_m - 540.0)
            + 4.0 * (0.20 - albedo)
        )
        lst_noise = self.rng.normal(0.0, 0.55, size=n)
        lst_celsius = np.clip(lst_base + lst_noise, 29.5, 45.5)

        # 8. Demographic & Socioeconomic Indicators
        # Population density (persons / km²): 4,000 to 48,000
        pop_density = np.clip(
            (
                core_factor * 28000.0
                + swargate_factor * 22000.0
                + 8000.0
                + self.rng.normal(0, 2500, size=n)
            ),
            2500.0,
            48000.0,
        )
        # Cell population (100m x 100m = 0.01 km²)
        cell_population = np.round(pop_density * 0.01).astype(int)

        # Vulnerable population ratio (elderly >65 + kids <5 + low-income/slum ratio)
        vuln_ratio = np.clip(
            0.22 + 0.28 * core_factor + 0.20 * swargate_factor + self.rng.normal(0, 0.05, size=n),
            0.10,
            0.85,
        )

        # Distance to cooling amenities (drinking water / emergency shelter) in meters
        dist_cooling_facility = np.clip(
            dist_river * 0.4 + dist_core * 0.3 + self.rng.exponential(400.0, size=n),
            50.0,
            2500.0,
        )

        # 9. Heat Vulnerability Index (HVI: 0 to 1)
        # Formulated as: 0.40 * Exposure (LST normalized) + 0.35 * Sensitivity (Vuln & Pop) + 0.25 * (1 - Adaptive Capacity)
        lst_norm = (lst_celsius - np.min(lst_celsius)) / (np.max(lst_celsius) - np.min(lst_celsius))
        sens_norm = 0.6 * vuln_ratio + 0.4 * (pop_density / np.max(pop_density))
        adapt_norm = 0.5 * (canopy_pct / 100.0) + 0.5 * (
            1.0 - np.clip(dist_cooling_facility / 2000.0, 0.0, 1.0)
        )

        hvi = 0.40 * lst_norm + 0.35 * sens_norm + 0.25 * (1.0 - adapt_norm)
        hvi = np.clip(hvi, 0.05, 0.98)

        # 10. Critical Hotspot Classification
        # Hotspots = Upper quartile of LST AND high vulnerability (top 15% combined severity)
        hotspot_threshold = np.percentile(lst_celsius, 82)
        is_critical_hotspot = (lst_celsius >= hotspot_threshold) & (hvi >= 0.62)

        # Transit presence flag (near station or bus stops)
        is_transit_hub = (dist_station < 500.0) | (dist_swargate < 500.0)

        # Assign attributes to projected GeoDataFrame
        gdf_proj["ndbi"] = np.round(ndbi, 4)
        gdf_proj["ndvi"] = np.round(ndvi, 4)
        gdf_proj["tree_canopy_pct"] = np.round(canopy_pct, 2)
        gdf_proj["impervious_pct"] = np.round(impervious_pct, 2)
        gdf_proj["albedo"] = np.round(albedo, 4)
        gdf_proj["elevation_m"] = np.round(elevation_m, 1)
        gdf_proj["lst_celsius"] = np.round(lst_celsius, 2)
        gdf_proj["lst_anomaly_celsius"] = np.round(lst_celsius - np.mean(lst_celsius), 2)
        gdf_proj["pop_density_sqkm"] = np.round(pop_density, 0).astype(int)
        gdf_proj["population"] = cell_population
        gdf_proj["vuln_population_ratio"] = np.round(vuln_ratio, 4)
        gdf_proj["dist_cooling_facility_m"] = np.round(dist_cooling_facility, 1)
        gdf_proj["heat_vulnerability_index"] = np.round(hvi, 4)
        gdf_proj["is_critical_hotspot"] = is_critical_hotspot.astype(bool)
        gdf_proj["is_transit_hub"] = is_transit_hub.astype(bool)

        # Explicit synthetic metadata label
        gdf_proj["is_synthetic"] = True
        gdf_proj["city"] = "Pune"
        gdf_proj["resolution_m"] = self.resolution

        # Reproject polygons to WGS84 for GeoJSON / Leaflet standard export
        gdf_wgs84 = gdf_proj.to_crs(self.crs_geo)

        # Remove projected intermediate coordinates before serialization
        gdf_wgs84 = gdf_wgs84.drop(columns=["x_proj", "y_proj"])
        return gdf_wgs84
