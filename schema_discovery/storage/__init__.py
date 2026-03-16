# schema_discovery/storage/__init__.py

from .duckdb_store import DuckDBTableStore

__all__ = ["DuckDBTableStore"]