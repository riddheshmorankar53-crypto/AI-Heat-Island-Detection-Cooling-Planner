"""
Feature engineering and dataset preparation for Urban Cooling Planner.

Adheres strictly to ml-best-practices:
- Train/Validation/Test split performed BEFORE fitting any scaling or transformations.
- Complete type hints and docstrings.
- Reproducible random seed.
"""

from typing import List, Tuple

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS: List[str] = [
    "ndbi",
    "ndvi",
    "tree_canopy_pct",
    "impervious_pct",
    "albedo",
    "elevation_m",
    "pop_density_sqkm",
    "vuln_population_ratio",
    "dist_cooling_facility_m",
]

TARGET_REGRESSION: str = "lst_celsius"
TARGET_CLASSIFICATION: str = "is_critical_hotspot"
SPATIAL_BLOCK_M: float = 500.0


def spatial_block_ids(df: pd.DataFrame, block_m: float = SPATIAL_BLOCK_M) -> np.ndarray:
    """Assign each cell to a square block (UTM 43N metres) from its centroid."""
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    x, y = to_utm.transform(df["centroid_lon"].to_numpy(), df["centroid_lat"].to_numpy())
    bx, by = np.floor(x / block_m).astype(int), np.floor(y / block_m).astype(int)
    return bx * 100_000 + by


class SpatialFeaturePipeline:
    """Manages feature extraction, scaling, and train/val/test splits."""

    def __init__(self, seed: int = 42) -> None:
        """
        Initialize feature pipeline.

        Args:
            seed: Random seed for reproducible splits.
        """
        self.seed = seed
        self.scaler = StandardScaler()
        self.feature_columns = FEATURE_COLUMNS
        self.is_fitted = False

    def prepare_splits(
        self,
        gdf: gpd.GeoDataFrame,
        test_size: float = 0.15,
        val_size: float = 0.15,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
    ]:
        """
        Splits data into train, validation, and test partitions BEFORE scaling.

        Returns:
            X_train, X_val, X_test, y_train_reg, y_val_reg, y_test_reg,
            y_train_cls, y_val_cls, y_test_cls
        """
        df = (
            pd.DataFrame(gdf.drop(columns=["geometry"]))
            if "geometry" in gdf.columns
            else gdf.copy()
        )

        # Handle any missing or invalid rows contextually
        for col in self.feature_columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        X = df[self.feature_columns]
        y_reg = df[TARGET_REGRESSION]
        y_cls = df[TARGET_CLASSIFICATION].astype(int)

        # Spatial block split: whole 500m blocks go to one partition, so neighbouring
        # (spatially autocorrelated) cells never straddle train and test.
        groups = spatial_block_ids(df)
        val_ratio_adjusted = val_size / (1.0 - test_size)
        trainval_idx, test_idx = next(
            GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=self.seed).split(
                X, groups=groups
            )
        )
        train_pos, val_pos = next(
            GroupShuffleSplit(
                n_splits=1, test_size=val_ratio_adjusted, random_state=self.seed
            ).split(trainval_idx, groups=groups[trainval_idx])
        )
        train_idx, val_idx = trainval_idx[train_pos], trainval_idx[val_pos]

        X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
        y_reg_train, y_reg_val, y_reg_test = (
            y_reg.iloc[train_idx],
            y_reg.iloc[val_idx],
            y_reg.iloc[test_idx],
        )
        y_cls_train, y_cls_val, y_cls_test = (
            y_cls.iloc[train_idx],
            y_cls.iloc[val_idx],
            y_cls.iloc[test_idx],
        )

        return (
            X_train,
            X_val,
            X_test,
            y_reg_train,
            y_reg_val,
            y_reg_test,
            y_cls_train,
            y_cls_val,
            y_cls_test,
        )

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit scaler on training features only and transform."""
        X_scaled = self.scaler.fit_transform(X_train)
        self.is_fitted = True
        return X_scaled

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform test or validation features using fitted scaler."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted on training data before transforming.")
        return self.scaler.transform(X)

    def save_scaler(self, path: str) -> None:
        """Persist fitted scaler artifact."""
        joblib.dump(self.scaler, path)

    def load_scaler(self, path: str) -> None:
        """Load persisted scaler artifact."""
        self.scaler = joblib.load(path)
        self.is_fitted = True
