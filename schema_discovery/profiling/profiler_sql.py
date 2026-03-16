# schema_discovery/profiling/profiler_sql.py

from __future__ import annotations

from typing import Any

import pandas as pd

from schema_discovery.storage.duckdb_store import DuckDBTableStore


def _quote_ident(name: str) -> str:
    """
    Quote DuckDB identifiers safely.
    """
    if not isinstance(name, str) or name == "":
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return '"' + name.replace('"', '""') + '"'


def _map_duckdb_type_to_family(dtype: str) -> str:
    """
    Minimal dtype mapping for relationship discovery.
    """
    d = str(dtype).upper()

    if "INT" in d:
        return "int"

    if any(tok in d for tok in ("DOUBLE", "FLOAT", "REAL", "DECIMAL")):
        return "float"

    if any(tok in d for tok in ("DATE", "TIME", "TIMESTAMP")):
        return "datetime"

    if "BOOL" in d:
        return "bool"

    if any(tok in d for tok in ("VARCHAR", "CHAR", "TEXT", "STRING")):
        return "string"

    return "other"


def _build_profile_sql(table_name: str, column_name: str) -> str:
    """
    Per column profiling query.

    Keeps the profiler simple and memory stable:
    - no giant wide query
    - no sample extraction
    - computes only the signals the pipeline needs
    """
    table_q = _quote_ident(table_name)
    col_q = _quote_ident(column_name)

    return f"""
    WITH clean AS (
        SELECT
            CASE
                WHEN {col_q} IS NULL THEN NULL
                WHEN lower(trim(CAST({col_q} AS VARCHAR))) IN ('', 'null', 'none', 'n/a', 'na', 'nan') THEN NULL
                ELSE trim(CAST({col_q} AS VARCHAR))
            END AS val
        FROM {table_q}
    ),
    freq AS (
        SELECT val, COUNT(*) AS cnt
        FROM clean
        WHERE val IS NOT NULL
        GROUP BY val
    ),
    nn AS (
        SELECT COUNT(*) AS non_null_count
        FROM clean
        WHERE val IS NOT NULL
    )
    SELECT
        COUNT(*) AS n_rows,
        SUM(CASE WHEN val IS NULL THEN 1 ELSE 0 END) AS n_null,
        SUM(CASE WHEN val IS NOT NULL THEN 1 ELSE 0 END) AS n_non_null,
        COUNT(DISTINCT val) FILTER (WHERE val IS NOT NULL) AS n_unique,
        SUM(CASE WHEN val IS NULL THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) AS null_ratio,
        COUNT(DISTINCT val) FILTER (WHERE val IS NOT NULL) * 1.0 / NULLIF(COUNT(*), 0) AS unique_ratio,
        COUNT(DISTINCT val) FILTER (WHERE val IS NOT NULL) * 1.0 / NULLIF(SUM(CASE WHEN val IS NOT NULL THEN 1 ELSE 0 END), 0) AS unique_ratio_non_null,
        COALESCE((SELECT MAX(cnt) * 1.0 / NULLIF(SUM(cnt), 0) FROM freq), 0.0) AS top1_ratio,
        AVG(LENGTH(val)) FILTER (WHERE val IS NOT NULL) AS avg_len,
        MIN(LENGTH(val)) FILTER (WHERE val IS NOT NULL) AS min_len,
        MAX(LENGTH(val)) FILTER (WHERE val IS NOT NULL) AS max_len
    FROM clean
    """


def profile_table_sql(
    store: DuckDBTableStore,
    table_name: str,
    *,
    sample_k: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Profile one table using DuckDB SQL.

    This version is intentionally lean:
    - one query per column
    - no sample values
    - no wide query explosion
    """
    desc = store.describe_table(table_name)

    if "column_name" not in desc.columns or "column_type" not in desc.columns:
        raise ValueError(f"Unexpected DESCRIBE output for table {table_name}")

    rows: list[dict[str, Any]] = []
    n_rows_table = store.count_rows(table_name)

    for _, meta in desc.iterrows():
        column_name = str(meta["column_name"])
        column_type = str(meta["column_type"])
        dtype_family = _map_duckdb_type_to_family(column_type)

        stats_sql = _build_profile_sql(table_name, column_name)
        stats_df = store.fetchdf(stats_sql)
        if stats_df.empty:
            continue

        srow = stats_df.iloc[0].to_dict()

        n_rows = int(srow.get("n_rows") or 0)
        n_null = int(srow.get("n_null") or 0)
        n_non_null = int(srow.get("n_non_null") or 0)
        n_unique = int(srow.get("n_unique") or 0)

        null_ratio = float(srow.get("null_ratio") or 0.0)
        unique_ratio = float(srow.get("unique_ratio") or 0.0)
        unique_ratio_non_null = float(srow.get("unique_ratio_non_null") or 0.0)
        top1_ratio = float(srow.get("top1_ratio") or 0.0)

        avg_len = srow.get("avg_len")
        min_len = srow.get("min_len")
        max_len = srow.get("max_len")

        is_unary_ucc = (n_rows > 0) and (n_null == 0) and (n_unique == n_rows)

        rows.append(
            {
                "table_name": str(table_name),
                "column_name": column_name,
                "column_key": f"{table_name}.{column_name}",
                "dtype": column_type,
                "dtype_family": dtype_family,
                "n_rows": int(n_rows),
                "n_null": int(n_null),
                "n_non_null": int(n_non_null),
                "null_ratio": float(null_ratio),
                "n_unique": int(n_unique),
                "unique_ratio": float(unique_ratio),
                "unique_ratio_non_null": float(unique_ratio_non_null),
                "is_unary_ucc": bool(is_unary_ucc),
                "avg_len": float(avg_len) if avg_len is not None else None,
                "min_len": float(min_len) if min_len is not None else None,
                "max_len": float(max_len) if max_len is not None else None,
                "top1_ratio": float(top1_ratio),
            }
        )

    profiles_df = pd.DataFrame(rows)
    if profiles_df.empty:
        return profiles_df, {
            "table_name": str(table_name),
            "n_rows": int(n_rows_table),
            "n_cols": 0,
        }

    profiles_df = (
        profiles_df
        .sort_values(["table_name", "column_name"], kind="mergesort")
        .reset_index(drop=True)
    )

    extras = {
        "table_name": str(table_name),
        "n_rows": int(n_rows_table),
        "n_cols": int(len(rows)),
    }

    return profiles_df, extras


def profile_all_tables_sql(
    store: DuckDBTableStore,
    *,
    sample_k: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Profile all registered tables in DuckDB.

    sample_k is ignored and kept only so existing call sites do not break.
    """
    frames: list[pd.DataFrame] = []
    extras: dict[str, Any] = {"tables": {}}

    for table_name in store.list_tables():
        t_profiles, t_extras = profile_table_sql(store, table_name, sample_k=sample_k)
        frames.append(t_profiles)
        extras["tables"][str(table_name)] = t_extras

    if frames:
        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values(["table_name", "column_name"], kind="mergesort").reset_index(drop=True)
    else:
        out = pd.DataFrame()

    return out, extras