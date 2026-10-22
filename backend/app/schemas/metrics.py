"""Schemas for ML evaluation metrics and transparency reporting."""

from typing import Any, Dict

from pydantic import BaseModel


class MLMetricsResponse(BaseModel):
    metadata: Dict[str, Any]
    models: Dict[str, Any]
