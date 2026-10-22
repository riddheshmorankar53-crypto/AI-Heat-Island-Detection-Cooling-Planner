"""
Data Exporter for Pune Urban Cooling Planner.

Handles serialization to GeoParquet and GeoJSON, and manages PostGIS loading
with spatial (GIST) and attribute (B-tree) indexes.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import geopandas as gpd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


class DataExporter:
    """Exports spatial datasets to GeoParquet, GeoJSON, and PostGIS."""

    def __init__(self, default_output_dir: Union[str, Path] = "data/processed") -> None:
        """
        Initialize exporter.

        Args:
            default_output_dir: Default directory for file exports.
        """
        self.output_dir = Path(default_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_geoparquet(
        self,
        gdf: gpd.GeoDataFrame,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Export GeoDataFrame to GeoParquet format.

        Args:
            gdf: GeoPandas GeoDataFrame.
            output_path: Target path for the parquet file.

        Returns:
            Resolved Path of the written file.
        """
        if output_path is None:
            output_path = self.output_dir / "pune_100m_grid.parquet"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to geoparquet using pyarrow engine
        gdf.to_parquet(output_path, engine="pyarrow", index=False)
        logger.info("Saved GeoParquet (%d records) to %s", len(gdf), output_path)
        return output_path

    def save_geojson(
        self,
        gdf: gpd.GeoDataFrame,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Export GeoDataFrame to GeoJSON format.

        Args:
            gdf: GeoPandas GeoDataFrame.
            output_path: Target path for the geojson file.

        Returns:
            Resolved Path of the written file.
        """
        if output_path is None:
            output_path = self.output_dir / "pune_100m_grid.geojson"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_path, driver="GeoJSON")
        logger.info("Saved GeoJSON (%d records) to %s", len(gdf), output_path)
        return output_path

    def load_postgis(
        self,
        gdf: gpd.GeoDataFrame,
        db_url: Optional[str] = None,
        table_name: str = "pune_100m_grid",
        if_exists: str = "replace",
    ) -> bool:
        """
        Load GeoDataFrame into a PostGIS table and create spatial + B-tree indexes.

        Args:
            gdf: GeoPandas GeoDataFrame.
            db_url: SQLAlchemy database URL (e.g. postgresql://user:pass@host:5432/db).
            table_name: Target PostGIS table name.
            if_exists: Action if table exists ('replace', 'append', 'fail').

        Returns:
            True if ingestion succeeded, False if unavailable or error.
        """
        if not db_url:
            logger.info("No PostGIS DB URL configured; skipping PostGIS ingestion.")
            return False

        try:
            engine = create_engine(db_url, connect_args={"connect_timeout": 3})
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            logger.info(
                "Connecting to PostGIS at %s to load %s...", db_url.split("@")[-1], table_name
            )
            gdf.to_postgis(table_name, engine, if_exists=if_exists, index=False)

            # Build spatial and attribute indexes
            with engine.connect() as conn:
                # Spatial GIST index on geometry
                conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_geom ON {table_name} USING GIST (geometry);"
                    )
                )
                # B-Tree index on cell_id
                if "cell_id" in gdf.columns:
                    conn.execute(
                        text(
                            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_cell_id ON {table_name} (cell_id);"
                        )
                    )
                # B-Tree index on lst_celsius / mean_lst
                lst_col = (
                    "lst_celsius"
                    if "lst_celsius" in gdf.columns
                    else ("mean_lst" if "mean_lst" in gdf.columns else None)
                )
                if lst_col:
                    conn.execute(
                        text(
                            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_lst ON {table_name} ({lst_col});"
                        )
                    )
                # B-Tree index on heat_vulnerability_index
                if "heat_vulnerability_index" in gdf.columns:
                    conn.execute(
                        text(
                            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_hvi ON {table_name} (heat_vulnerability_index);"
                        )
                    )
                conn.commit()

            logger.info(
                "Successfully loaded %d records into PostGIS table '%s' with GIST and B-tree indexes.",
                len(gdf),
                table_name,
            )
            return True

        except Exception as exc:
            logger.warning(
                "PostGIS connection or insertion skipped (PostGIS may be offline): %s", exc
            )
            return False

    def export_all(
        self,
        gdf: gpd.GeoDataFrame,
        parquet_path: Optional[Union[str, Path]] = None,
        geojson_path: Optional[Union[str, Path]] = None,
        db_url: Optional[str] = None,
        table_name: str = "pune_100m_grid",
    ) -> Dict[str, Any]:
        """
        Execute full export to GeoParquet, GeoJSON, and PostGIS.

        Args:
            gdf: GeoPandas GeoDataFrame.
            parquet_path: Path for GeoParquet export.
            geojson_path: Path for GeoJSON export.
            db_url: Optional database URL.
            table_name: PostGIS table name.

        Returns:
            Dictionary with export status and paths.
        """
        parquet_file = self.save_geoparquet(gdf, parquet_path)
        geojson_file = self.save_geojson(gdf, geojson_path)
        postgis_loaded = self.load_postgis(gdf, db_url=db_url, table_name=table_name)

        return {
            "parquet_path": str(parquet_file),
            "geojson_path": str(geojson_file),
            "postgis_loaded": postgis_loaded,
            "record_count": len(gdf),
        }
