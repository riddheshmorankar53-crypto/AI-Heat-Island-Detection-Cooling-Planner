"""
Backend application settings loaded from environment and config.yaml.
"""

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application configuration with zero hard-coded values."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Urban Cooling Planner API"
    version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    port: int = 8000
    host: str = "0.0.0.0"
    cors_origins: List[str] = ["http://localhost:5173", "http://127.0.0.1:5173", "*"]

    # Data paths
    grid_geojson_path: str = str(BASE_DIR / "data" / "processed" / "pune_100m_grid.geojson")
    street_network_path: str = str(BASE_DIR / "data" / "processed" / "pune_pedestrian_network.json")
    metrics_json_path: str = str(BASE_DIR / "ml" / "artifacts" / "metrics.json")
    aoi_yaml_path: str = str(BASE_DIR / "config" / "pune_aoi.yaml")
    config_yaml_path: str = str(BASE_DIR / "config" / "config.yaml")


settings = Settings()
