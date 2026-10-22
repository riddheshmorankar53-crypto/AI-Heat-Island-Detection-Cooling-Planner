"""
Satellite imagery ingestion via Microsoft Planetary Computer STAC.

Builds cloud-masked median composites on the 100m grid for `python data/pipeline.py --live`:
1. Landsat 8/9 Collection 2 Level-2 surface temperature (LST).
2. Sentinel-2 Level-2A NDVI, NDBI, broadband albedo, canopy % and impervious %.
3. Copernicus GLO-30 elevation.
Only the AOI window of each cloud-optimised GeoTIFF is read.
"""

import warnings
from typing import Any, Dict, List

import numpy as np
import planetary_computer
import pystac_client

STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

GRID_CRS = "EPSG:32643"  # UTM 43N, same as the fishnet
FINE_M = 10.0  # working resolution; 10 x 10 fine pixels per 100m cell
CELL_M = 100.0
REAL_DATE_RANGE = "2024-03-01/2024-05-31"  # Pune pre-monsoon peak heat
MAX_SCENES = 6
S2_VALID_SCL = (4, 5, 6, 7)  # vegetation, bare soil, water, unclassified
# ponytail: NDVI/NDBI thresholds are proxies for canopy and impervious cover;
# swap in a canopy-height or built-up product (e.g. ESA WorldCover) for accuracy.
TREE_NDVI = 0.4
BUILT_NDBI = 0.0
REAL_COLUMNS = [
    "lst_celsius",
    "ndvi",
    "ndbi",
    "albedo",
    "tree_canopy_pct",
    "impervious_pct",
    "elevation_m",
]


def grid_layout(gdf) -> Dict[str, Any]:
    """Fine raster layout aligned to the fishnet, plus each cell's (row, col) block."""
    from rasterio.transform import from_origin

    centroids = gdf.to_crs(GRID_CRS).geometry.centroid
    cx, cy = centroids.x.to_numpy(), centroids.y.to_numpy()
    cols = np.rint((cx - cx.min()) / CELL_M).astype(int)
    rows = np.rint((cy.max() - cy) / CELL_M).astype(int)
    k = int(CELL_M / FINE_M)
    return {
        "transform": from_origin(cx.min() - CELL_M / 2, cy.max() + CELL_M / 2, FINE_M, FINE_M),
        "shape": ((rows.max() + 1) * k, (cols.max() + 1) * k),
        "rows": rows,
        "cols": cols,
        "k": k,
    }


def cell_values(fine: np.ndarray, layout: Dict[str, Any]) -> np.ndarray:
    """Average a fine raster into one value per grid cell (NaN pixels ignored)."""
    k = layout["k"]
    h, w = fine.shape
    blocks = fine.reshape(h // k, k, w // k, k).transpose(0, 2, 1, 3).reshape(h // k, w // k, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks -> NaN
        per_block = np.nanmean(blocks, axis=2)
    return per_block[layout["rows"], layout["cols"]]


def _nanmedian(stack: List[np.ndarray]) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.stack(stack), axis=0)


def _read_to_grid(href: str, layout: Dict[str, Any], nearest: bool = False) -> np.ndarray:
    """Read only the AOI window of a remote COG and warp it onto the fine grid (NaN = no data)."""
    import rasterio
    from rasterio.errors import WindowError
    from rasterio.transform import array_bounds
    from rasterio.warp import Resampling, reproject, transform_bounds
    from rasterio.windows import Window, from_bounds

    dst = np.full(layout["shape"], np.nan, dtype=np.float32)
    bounds = array_bounds(*layout["shape"], layout["transform"])
    with rasterio.open(href) as src:
        win = from_bounds(*transform_bounds(GRID_CRS, src.crs, *bounds), src.transform)
        win = Window(win.col_off - 2, win.row_off - 2, win.width + 4, win.height + 4)
        try:
            win = (
                win.round_offsets()
                .round_lengths()
                .intersection(Window(0, 0, src.width, src.height))
            )
        except WindowError:
            return dst  # scene does not cover the AOI
        data = src.read(1, window=win).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        reproject(
            data,
            dst,
            src_transform=src.window_transform(win),
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=layout["transform"],
            dst_crs=GRID_CRS,
            dst_nodata=np.nan,
            resampling=Resampling.nearest if nearest else Resampling.bilinear,
        )
    return dst


def _search(collection: str, bbox: List[float], **kwargs) -> List[Any]:
    client = pystac_client.Client.open(STAC_API_URL, modifier=planetary_computer.sign_inplace)
    return list(client.search(collections=[collection], bbox=bbox, **kwargs).items())


def _least_cloudy(items: List[Any]) -> List[Any]:
    return sorted(items, key=lambda i: i.properties.get("eo:cloud_cover", 100))[:MAX_SCENES]


def landsat_lst(bbox: List[float], layout: Dict[str, Any]) -> np.ndarray:
    """Cloud-masked median Landsat 8/9 surface temperature (°C) on the fine grid."""
    items = _search(
        "landsat-c2-l2",
        bbox,
        datetime=REAL_DATE_RANGE,
        query={"eo:cloud_cover": {"lt": 20}, "platform": {"in": ["landsat-8", "landsat-9"]}},
    )
    scenes = []
    for item in _least_cloudy(items):
        st = _read_to_grid(item.assets["lwir11"].href, layout)
        qa = _read_to_grid(item.assets["qa_pixel"].href, layout, nearest=True)
        # QA_PIXEL bits 0-4: fill, dilated cloud, cirrus, cloud, cloud shadow
        bad = np.isnan(qa) | ((np.nan_to_num(qa).astype(np.uint16) & 0b11111) != 0)
        st[bad] = np.nan
        scenes.append(st * 0.00341802 + 149.0 - 273.15)  # Collection 2 ST scale -> °C
    if not scenes:
        raise RuntimeError("No Landsat 8/9 scenes found for the AOI and date range.")
    print(f"Landsat LST: median of {len(scenes)} scenes")
    return _nanmedian(scenes)


def sentinel2_indices(bbox: List[float], layout: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Cloud-masked median NDVI, NDBI and broadband albedo from Sentinel-2 L2A."""
    items = _search(
        "sentinel-2-l2a", bbox, datetime=REAL_DATE_RANGE, query={"eo:cloud_cover": {"lt": 10}}
    )
    ndvi_s, ndbi_s, albedo_s = [], [], []
    for item in _least_cloudy(items):
        # Processing baseline >= 04.00 adds a +1000 DN offset to reflectances
        offset = 1000.0 if float(item.properties.get("s2:processing_baseline", "0")) >= 4.0 else 0
        b = {
            name: (_read_to_grid(item.assets[name].href, layout) - offset) / 10000.0
            for name in ("B02", "B03", "B04", "B08", "B11", "B12")
        }
        bad = ~np.isin(_read_to_grid(item.assets["SCL"].href, layout, nearest=True), S2_VALID_SCL)
        with np.errstate(all="ignore"):
            ndvi = (b["B08"] - b["B04"]) / (b["B08"] + b["B04"])
            ndbi = (b["B11"] - b["B08"]) / (b["B11"] + b["B08"])
        # Bonafoni & Sekertekin (2020) Sentinel-2 broadband albedo
        albedo = (
            0.2266 * b["B02"]
            + 0.1236 * b["B03"]
            + 0.1573 * b["B04"]
            + 0.3417 * b["B08"]
            + 0.1170 * b["B11"]
            + 0.0338 * b["B12"]
        )
        for arr, out in ((ndvi, ndvi_s), (ndbi, ndbi_s), (albedo, albedo_s)):
            arr[bad] = np.nan
            out.append(arr)
    if not ndvi_s:
        raise RuntimeError("No Sentinel-2 L2A scenes found for the AOI and date range.")
    print(f"Sentinel-2 indices: median of {len(ndvi_s)} scenes")
    return {"ndvi": _nanmedian(ndvi_s), "ndbi": _nanmedian(ndbi_s), "albedo": _nanmedian(albedo_s)}


def copernicus_dem(bbox: List[float], layout: Dict[str, Any]) -> np.ndarray:
    """Copernicus GLO-30 elevation (m) on the fine grid, mosaicking tiles."""
    dem = np.full(layout["shape"], np.nan, dtype=np.float32)
    for item in _search("cop-dem-glo-30", bbox):
        dem = np.where(np.isnan(dem), _read_to_grid(item.assets["data"].href, layout), dem)
    return dem


def apply_real_satellite(gdf, bbox: List[float]):
    """
    Replace the satellite-derivable grid columns with real observations:
    LST (Landsat 8/9), NDVI / NDBI / albedo / canopy % / impervious % (Sentinel-2 L2A),
    elevation (Copernicus DEM). Call `recompute_derived` afterwards for HVI and hotspots.
    """
    layout = grid_layout(gdf)
    lst = landsat_lst(bbox, layout)
    s2 = sentinel2_indices(bbox, layout)
    dem = copernicus_dem(bbox, layout)

    valid = ~np.isnan(s2["ndvi"])

    def pct(mask: np.ndarray) -> np.ndarray:
        return cell_values(np.where(valid, mask.astype(np.float32), np.nan), layout) * 100.0

    out = gdf.copy()
    out["lst_celsius"] = cell_values(lst, layout)
    out["ndvi"] = cell_values(s2["ndvi"], layout)
    out["ndbi"] = cell_values(s2["ndbi"], layout)
    out["albedo"] = cell_values(s2["albedo"], layout)
    out["tree_canopy_pct"] = pct(s2["ndvi"] > TREE_NDVI)
    out["impervious_pct"] = pct((s2["ndbi"] > BUILT_NDBI) & (s2["ndvi"] < 0.2))
    out["elevation_m"] = cell_values(dem, layout)

    for col in REAL_COLUMNS:
        missing = int(out[col].isna().sum())
        if missing:
            print(f"WARNING {col}: {missing} cells cloud-covered in every scene, median-filled")
            out[col] = out[col].fillna(out[col].median())
    decimals = {"lst_celsius": 2, "ndvi": 4, "ndbi": 4, "albedo": 4, "elevation_m": 1}
    out = out.round({c: decimals.get(c, 2) for c in REAL_COLUMNS})

    out["satellite_source"] = "landsat-c2-l2 + sentinel-2-l2a + cop-dem-glo-30"
    return out
