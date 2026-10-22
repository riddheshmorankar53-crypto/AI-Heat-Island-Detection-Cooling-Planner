"""
Gradient Boosted Decision Tree models (LightGBM / XGBoost) for
Land Surface Temperature regression and Critical Hotspot classification.
"""

from typing import Any, Dict, List, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


class GradientBoostLSTModel:
    """LightGBM regressor predicting 100m grid cell Land Surface Temperature (LST)."""

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        random_state: int = 42,
    ) -> None:
        self.random_state = random_state
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            verbosity=-1,
            n_jobs=-1,
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "GradientBoostLSTModel":
        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        self.model.fit(X_train, y_train, eval_set=eval_set)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        preds = self.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
        }

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        importances = self.model.feature_importances_
        norm_imp = importances / np.sum(importances)
        return {name: round(float(val), 4) for name, val in zip(feature_names, norm_imp)}

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)


class GradientBoostHotspotClassifier:
    """LightGBM classifier identifying Critical Urban Heat Island Hotspots."""

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        random_state: int = 42,
    ) -> None:
        self.random_state = random_state
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",
            verbosity=-1,
            n_jobs=-1,
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "GradientBoostHotspotClassifier":
        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        self.model.fit(X_train, y_train, eval_set=eval_set)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        preds = self.predict(X_test)
        cm = confusion_matrix(y_test, preds).tolist()
        return {
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, preds, zero_division=0)), 4),
            "confusion_matrix": cm,
        }

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        importances = self.model.feature_importances_
        norm_imp = importances / np.sum(importances)
        return {name: round(float(val), 4) for name, val in zip(feature_names, norm_imp)}

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
