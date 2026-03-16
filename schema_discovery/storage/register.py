# schema_discovery/storage/register.py

from __future__ import annotations

from pathlib import Path

from schema_discovery.storage.duckdb_store import DuckDBTableStore


def register_sources(
    store: DuckDBTableStore,
    table_sources: dict[str, str | Path],
    *,
    csv_all_varchar: bool = False,
) -> None:
    """
    Register all source files as DuckDB views.

    Example:
        {
            "clients_df": "data/clients.parquet",
            "payments_df": "data/payments.csv",
        }
    """
    for table_name, path in table_sources.items():
        store.register_path(
            table_name=table_name,
            path=path,
            csv_all_varchar=csv_all_varchar,
        )