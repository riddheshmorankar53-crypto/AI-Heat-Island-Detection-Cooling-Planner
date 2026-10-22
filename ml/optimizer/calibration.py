"""
Measure intervention cooling effects from the real grid instead of assuming them.

Only tree canopy is identifiable here: cells are compared with others in the same 500m
block (block fixed effects), controlling for impervious cover and elevation, so the
core-vs-suburb contrast does not masquerade as a tree effect. Albedo is not identifiable
from a cross-section (Pune's brightest surfaces are hot bare soil, not cool roofs), and
misting / shade / refuges cool people rather than the land surface Landsat measures.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from ml.features.builder import spatial_block_ids

CONTROLS = ["impervious_pct", "elevation_m"]


def _within_block(df: pd.DataFrame, cols: list, blocks: np.ndarray) -> np.ndarray:
    """Subtract each block's mean (block fixed effects)."""
    return (df[cols] - df[cols].groupby(blocks).transform("mean")).to_numpy()


def estimate_canopy_cooling(gdf: pd.DataFrame, n_boot: int = 300, seed: int = 42) -> Dict[str, Any]:
    """
    LST change (°C) per +1 percentage point of tree canopy, with a block-bootstrap 95% CI.
    Negative = cooling.
    """
    cols = ["tree_canopy_pct"] + CONTROLS
    blocks = spatial_block_ids(gdf)
    X = _within_block(gdf, cols, blocks)
    y = _within_block(gdf, ["lst_celsius"], blocks)[:, 0]
    coef = float(np.linalg.lstsq(X, y, rcond=None)[0][0])

    # Resample whole blocks, so the CI respects spatial autocorrelation
    unique = np.unique(blocks)
    rows_of = {b: np.flatnonzero(blocks == b) for b in unique}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([rows_of[b] for b in rng.choice(unique, len(unique))])
        boot.append(np.linalg.lstsq(X[idx], y[idx], rcond=None)[0][0])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return {
        "lst_change_per_canopy_pct": round(coef, 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "n_cells": int(len(gdf)),
        "method": "OLS with 500m-block fixed effects, controls: " + ", ".join(CONTROLS),
    }
