"""
Model Evaluation Metrics and Transparency Endpoints.
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.app.config import settings
from backend.app.schemas.metrics import MLMetricsResponse

router = APIRouter(prefix="/api/v1/ml", tags=["Model Transparency & Metrics"])


@router.get("/metrics", response_model=MLMetricsResponse)
def get_ml_metrics() -> MLMetricsResponse:
    """
    Retrieve genuine trained model evaluation numbers (R2, RMSE, MAE, Confusion Matrix).
    Strict adherence to rule: Never fabricate real-world metrics; report real evaluations only.
    """
    metrics_path = Path(settings.metrics_json_path)
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Model metrics artifact not found. Please execute `python ml/train.py` first.",
        )

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return MLMetricsResponse(**data)
