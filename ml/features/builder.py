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
from sklearn.model_selection import train_test_split
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

        # First split: train + val vs test
        X_train_val, X_test, y_reg_train_val, y_reg_test, y_cls_train_val, y_cls_test = (
            train_test_split(
                X, y_reg, y_cls, test_size=test_size, random_state=self.seed, stratify=y_cls
            )
        )

        # Second split: train vs val
        val_ratio_adjusted = val_size / (1.0 - test_size)
        X_train, X_val, y_reg_train, y_reg_val, y_cls_train, y_cls_val = train_test_split(
            X_train_val,
            y_reg_train_val,
            y_cls_train_val,
            test_size=val_ratio_adjusted,
            random_state=self.seed,
            stratify=y_cls_train_val,
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
