"""
Baseline models for Urban Heat Island regression and hotspot classification.
Used as benchmarks to compare against tree-based and graph neural networks.
"""

from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class BaselineLSTModel:
    """Ridge regression baseline for continuous Land Surface Temperature."""

    def __init__(self, alpha: float = 1.0, random_state: int = 42) -> None:
        self.model = Ridge(alpha=alpha, random_state=random_state)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BaselineLSTModel":
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


class BaselineHotspotClassifier:
    """Logistic regression baseline for binary critical hotspot detection."""

    def __init__(self, random_state: int = 42) -> None:
        self.model = LogisticRegression(
            class_weight="balanced", random_state=random_state, max_iter=500
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BaselineHotspotClassifier":
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        return {"accuracy": round(acc, 4), "f1_score": round(f1, 4)}
