"""
3-day air temperature forecast for Pune from Open-Meteo (free, no API key).
Turns static hotspot alerts into early warnings: a forecast heatwave escalates severity.
"""

import json
import time
import urllib.request
from typing import Any, Dict, Optional

OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
    "&daily=temperature_2m_max&timezone=Asia%2FKolkata&forecast_days=3"
)
PUNE_CENTER = (18.5180, 73.8560)  # lat, lon (Shaniwar Wada)
# IMD heatwave criteria for plains, simplified to absolute daily maxima
HEATWAVE_C = 40.0
SEVERE_HEATWAVE_C = 45.0
CACHE_SECONDS = 3600

_cache: Dict[str, Any] = {"at": 0.0, "value": None}


def fetch_forecast() -> Optional[Dict[str, Any]]:
    """Daily max temperatures for the next 3 days, cached for an hour. None if offline."""
    if _cache["value"] is not None and time.time() - _cache["at"] < CACHE_SECONDS:
        return _cache["value"]
    try:
        url = OPEN_METEO_URL.format(lat=PUNE_CENTER[0], lon=PUNE_CENTER[1])
        with urllib.request.urlopen(url, timeout=5) as resp:
            daily = json.load(resp)["daily"]
    except Exception as e:
        print(f"Open-Meteo forecast unavailable ({e}); alerts use satellite data only.")
        return None
    value = {
        "source": "open-meteo",
        "days": [
            {"date": d, "tmax_celsius": t}
            for d, t in zip(daily["time"], daily["temperature_2m_max"])
        ],
    }
    _cache.update(at=time.time(), value=value)
    return value


def escalation_for(forecast: Optional[Dict[str, Any]]) -> int:
    """Severity levels to add: +1 for a heatwave day (>= 40°C), +2 for a severe one (>= 45°C)."""
    if not forecast:
        return 0
    peak = max(d["tmax_celsius"] for d in forecast["days"])
    return int(peak >= HEATWAVE_C) + int(peak >= SEVERE_HEATWAVE_C)
