"""
100m x 100m Fishnet Grid Builder and Spatial Feature Engineering.

Constructs regular metric polygons over the Pune Pilot AOI (EPSG:32643 -> EPSG:4326)
and calculates per-cell microclimate, urban form, satellite, and demographic features:
- mean LST (°C)
- mean NDVI, NDBI, NDWI
- building density & mean building height
- sky view factor (SVF proxy)
- road density & impervious fraction
- distance to water bodies & distance to parks
- elevation (DEM) & population counts
- % children, % elderly, low-income proxy
- heat vulnerability index (HVI) & hotspot classification
"""

from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely.geometry import Point, Polygon

from data.ingestion.demographics import DemographicsIngestion
from data.ingestion.elevation_weather import ElevationWeatherIngestion
from data.ingestion.osm import OSMDataIngestion
from data.ingestion.satellite import SatelliteDataIngestion


class FishnetGridBuilder:
    """Builds 100m fishnet polygons and computes cell features."""

    def __init__(
        self,
        bbox: Dict[str, float],
        resolution_meters: float = 100.0,
        crs_geo: str = "EPSG:4326",
        crs_proj: str = "EPSG:32643",
        seed: int = 42,
    ) -> None:
        self.bbox = {
            "min_longitude": float(bbox.get("min_longitude", bbox.get("min_lon", 73.8250))),
            "min_latitude": float(bbox.get("min_latitude", bbox.get("min_lat", 18.4900))),
            "max_longitude": float(bbox.get("max_longitude", bbox.get("max_lon", 73.8850))),
            "max_latitude": float(bbox.get("max_latitude", bbox.get("max_lat", 18.5450))),
        }
        self.resolution = resolution_meters
        self.crs_geo = crs_geo
        self.crs_proj = crs_proj
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.to_proj = Transformer.from_crs(crs_geo, crs_proj, always_xy=True)
        self.to_geo = Transformer.from_crs(crs_proj, crs_geo, always_xy=True)

        # Ingestion sub-modules
        self.satellite_module = SatelliteDataIngestion()
        self.osm_module = OSMDataIngestion()
        self.elevation_module = ElevationWeatherIngestion()
        self.demographics_module = DemographicsIngestion()

        # Anchor points for Pune morphology (lon, lat)
        self.commercial_core = (73.8560, 18.5180)  # Budhwar Peth / Shaniwar Wada
        self.green_hill = (73.8310, 18.5280)  # Vetal Tekdi edge
        self.swargate = (73.8580, 18.5018)  # Swargate
        self.pune_station = (73.8744, 18.5284)  # Pune Station

    def generate_fishnet_cells(self) -> gpd.GeoDataFrame:
        """Create non-overlapping regular 100m square fishnet grid."""
        min_x, min_y = self.to_proj.transform(self.bbox["min_longitude"], self.bbox["min_latitude"])
        max_x, max_y = self.to_proj.transform(self.bbox["max_longitude"], self.bbox["max_latitude"])

        x_steps = np.arange(min_x, max_x, self.resolution)
        y_steps = np.arange(min_y, max_y, self.resolution)

        cells = []
        cell_ids = []
        x_centers = []
        y_centers = []

        counter = 0
        for x in x_steps:
            for y in y_steps:
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
                x_centers.append(x + self.resolution / 2.0)
                y_centers.append(y + self.resolution / 2.0)
                counter += 1

        gdf_proj = gpd.GeoDataFrame(
            {
                "cell_id": cell_ids,
                "x_proj": x_centers,
                "y_proj": y_centers,
            },
            geometry=cells,
            crs=self.crs_proj,
        )
        return gdf_proj

    def _distance_to_pt(self, x: np.ndarray, y: np.ndarray, pt: Tuple[float, float]) -> np.ndarray:
        pt_x, pt_y = self.to_proj.transform(pt[0], pt[1])
        return np.hypot(x - pt_x, y - pt_y)

    def compute_features(self, synthetic: bool = True) -> gpd.GeoDataFrame:
        """
        Build fishnet and calculate all features per 100m cell.

        Args:
            synthetic: If True, uses offline calibrated synthetic feature generator.

        Returns:
            GeoPandas GeoDataFrame in EPSG:4326 with complete feature set.
        """
        gdf_proj = self.generate_fishnet_cells()
        x = gdf_proj["x_proj"].to_numpy()
        y = gdf_proj["y_proj"].to_numpy()
        n = len(gdf_proj)

        # Reproject centroids to geographic coordinates
        lon_arr, lat_arr = self.to_geo.transform(x, y)
        gdf_proj["centroid_lon"] = np.round(lon_arr, 6)
        gdf_proj["centroid_lat"] = np.round(lat_arr, 6)

        # Distances to urban anchors
        dist_core = self._distance_to_pt(x, y, self.commercial_core)
        dist_hill = self._distance_to_pt(x, y, self.green_hill)
        dist_swargate = self._distance_to_pt(x, y, self.swargate)
        dist_station = self._distance_to_pt(x, y, self.pune_station)

        # 1. OSM vectors (Water, Parks, Amenities)
        bbox_dict = {
            "min_latitude": self.bbox["min_latitude"],
            "min_longitude": self.bbox["min_longitude"],
            "max_latitude": self.bbox["max_latitude"],
            "max_longitude": self.bbox["max_longitude"],
        }
        osm_layers = self.osm_module.fetch_or_generate(bbox_dict, synthetic=synthetic)

        # Distance to water bodies (Mutha river) in meters
        water_gdf_proj = osm_layers["water"].to_crs(self.crs_proj)
        water_union = water_gdf_proj.union_all()
        centroids_geom = [Point(px, py) for px, py in zip(x, y)]
        dist_water = np.array([water_union.distance(pt) for pt in centroids_geom])

        # Distance to parks in meters
        parks_gdf_proj = osm_layers["parks"].to_crs(self.crs_proj)
        parks_union = parks_gdf_proj.union_all()
        dist_park = np.array([parks_union.distance(pt) for pt in centroids_geom])

        # 2. Satellite rasters (LST, NDVI, NDBI, NDWI)
        bbox_list = [
            self.bbox["min_longitude"],
            self.bbox["min_latitude"],
            self.bbox["max_longitude"],
            self.bbox["max_latitude"],
        ]
        sat_data = self.satellite_module.fetch_or_generate(
            bbox=bbox_list,
            x_coords=x,
            y_coords=y,
            river_dist=dist_water,
            core_dist=dist_core,
            hill_dist=dist_hill,
            synthetic=synthetic,
        )

        # 3. Elevation (DEM in meters)
        elev_m = self.elevation_module.compute_elevation_profile(
            x_coords=x,
            y_coords=y,
            hill_dist=dist_hill,
            river_dist=dist_water,
            seed=self.seed,
        )

        # 4. Urban Form & Geometry Features
        # Building density: fraction of cell area (0.05 to 0.75)
        core_decay = np.exp(-dist_core / 1800.0)
        station_decay = np.exp(-dist_station / 1200.0)
        hill_decay = np.exp(-dist_hill / 1500.0)

        building_density = np.clip(
            0.15
            + 0.50 * core_decay
            + 0.15 * station_decay
            - 0.18 * hill_decay
            + self.rng.normal(0, 0.04, size=n),
            0.04,
            0.78,
        )

        # Mean building height (meters): 4.5m (1-2 floors) to 24m (7-8 floors in commercial areas)
        mean_bldg_height = np.clip(
            6.0
            + 14.0 * core_decay
            + 8.0 * station_decay
            - 3.0 * hill_decay
            + self.rng.normal(0, 1.2, size=n),
            3.5,
            26.0,
        )

        # Sky View Factor (SVF) proxy: fraction of visible sky in street canyon
        # SVF ~ 1.0 - 0.70 * density * (height / 25.0)
        svf_proxy = 1.0 - 0.70 * building_density * np.clip(mean_bldg_height / 25.0, 0.1, 1.0)
        svf_proxy = np.clip(svf_proxy + self.rng.normal(0, 0.02, size=n), 0.22, 0.96)

        # Road density: walkable road length in cell (meters per 10,000m² cell: 20m to 240m)
        road_density = np.clip(
            45.0 + 140.0 * core_decay + 50.0 * station_decay + self.rng.normal(0, 15.0, size=n),
            15.0,
            250.0,
        )

        # Impervious surface fraction: combined roof + asphalt pavement (0.10 to 0.95)
        impervious_fraction = np.clip(
            building_density + (road_density / 300.0) * 0.35 + self.rng.normal(0, 0.03, size=n),
            0.08,
            0.96,
        )

        # Tree canopy coverage (%)
        tree_canopy_pct = np.clip(
            (sat_data["mean_ndvi"] - 0.10) * 135.0 + self.rng.normal(0, 3.0, size=n), 2.0, 75.0
        )

        # 5. Demographics & Vulnerability Proxies
        demo_data = self.demographics_module.compute_demographic_features(
            core_dist=dist_core,
            swargate_dist=dist_swargate,
            station_dist=dist_station,
            seed=self.seed,
        )

        # 6. Composite Heat Vulnerability Index (HVI)
        # 0.40 * Exposure (LST normalized) + 0.35 * Sensitivity (demographics) + 0.25 * (1 - Adaptive Capacity)
        lst_arr = sat_data["mean_lst"]
        lst_norm = (lst_arr - np.min(lst_arr)) / (np.max(lst_arr) - np.min(lst_arr))
        sens_norm = (
            0.5 * (demo_data["pct_children"] / 15.0)
            + 0.5 * (demo_data["pct_elderly"] / 15.0)
            + 0.3 * demo_data["low_income_proxy"]
        )
        sens_norm = np.clip(sens_norm / 1.3, 0.05, 0.98)
        adapt_norm = 0.5 * (tree_canopy_pct / 100.0) + 0.5 * (
            1.0 - np.clip(dist_water / 1800.0, 0.0, 1.0)
        )

        hvi = 0.40 * lst_norm + 0.35 * sens_norm + 0.25 * (1.0 - adapt_norm)
        hvi = np.clip(hvi, 0.05, 0.98)

        # Critical Hotspot classification: upper quartile LST + high HVI
        lst_p82 = np.percentile(lst_arr, 82)
        is_critical_hotspot = (lst_arr >= lst_p82) & (hvi >= 0.62)

        # Populate GeoDataFrame
        gdf_proj["mean_lst"] = sat_data["mean_lst"]
        gdf_proj["lst_celsius"] = sat_data["mean_lst"]  # Compatibility alias
        gdf_proj["mean_ndvi"] = sat_data["mean_ndvi"]
        gdf_proj["ndvi"] = sat_data["mean_ndvi"]  # Compatibility alias
        gdf_proj["mean_ndbi"] = sat_data["mean_ndbi"]
        gdf_proj["ndbi"] = sat_data["mean_ndbi"]  # Compatibility alias
        gdf_proj["mean_ndwi"] = sat_data["mean_ndwi"]
        gdf_proj["building_density"] = np.round(building_density, 3)
        gdf_proj["mean_building_height"] = np.round(mean_bldg_height, 1)
        gdf_proj["sky_view_factor"] = np.round(svf_proxy, 3)
        gdf_proj["road_density_m"] = np.round(road_density, 1)
        gdf_proj["impervious_fraction"] = np.round(impervious_fraction, 3)
        gdf_proj["impervious_pct"] = np.round(impervious_fraction * 100.0, 1)  # Compatibility alias
        gdf_proj["tree_canopy_pct"] = np.round(tree_canopy_pct, 1)
        gdf_proj["dist_to_water_m"] = np.round(dist_water, 1)
        gdf_proj["dist_to_park_m"] = np.round(dist_park, 1)
        gdf_proj["elevation_m"] = elev_m
        gdf_proj["population"] = demo_data["population"]
        gdf_proj["pop_density_sqkm"] = demo_data["pop_density_sqkm"]
        gdf_proj["pct_children"] = demo_data["pct_children"]
        gdf_proj["pct_elderly"] = demo_data["pct_elderly"]
        gdf_proj["low_income_proxy"] = demo_data["low_income_proxy"]
        gdf_proj["heat_vulnerability_index"] = np.round(hvi, 4)
        gdf_proj["is_critical_hotspot"] = is_critical_hotspot.astype(bool)

        # Clear synthetic label & metadata
        gdf_proj["is_synthetic"] = True
        gdf_proj["city"] = "Pune"
        gdf_proj["resolution_m"] = self.resolution

        # Reproject to EPSG:4326 for final export
        gdf_wgs84 = gdf_proj.to_crs(self.crs_geo)
        gdf_wgs84 = gdf_wgs84.drop(columns=["x_proj", "y_proj"])
        return gdf_wgs84
