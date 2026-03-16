# schema_discovery/candidates/ind_unary_sql.py

from __future__ import annotations

import pandas as pd

from schema_discovery.storage.duckdb_store import DuckDBTableStore


def _quote_ident(name: str) -> str:
    """
    Quote DuckDB identifiers safely.

    Supports real CSV column names like:
    - Number/Email
    - Client ID
    - Address (Line 1)

    Escapes embedded double quotes by doubling them.
    """
    if not isinstance(name, str) or name == "":
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return '"' + name.replace('"', '""') + '"'


def _build_pair_metrics_sql(
    fk_table: str,
    fk_col: str,
    pk_table: str,
    pk_col: str,
) -> str:
    fk_table_q = _quote_ident(fk_table)
    fk_col_q = _quote_ident(fk_col)
    pk_table_q = _quote_ident(pk_table)
    pk_col_q = _quote_ident(pk_col)

    return f"""
    WITH fk_clean AS (
        SELECT
            CASE
                WHEN {fk_col_q} IS NULL THEN NULL
                WHEN lower(trim(CAST({fk_col_q} AS VARCHAR))) IN ('', 'null', 'none', 'n/a', 'na', 'nan') THEN NULL
                ELSE trim(CAST({fk_col_q} AS VARCHAR))
            END AS v
        FROM {fk_table_q}
    ),
    pk_clean AS (
        SELECT DISTINCT
            CASE
                WHEN {pk_col_q} IS NULL THEN NULL
                WHEN lower(trim(CAST({pk_col_q} AS VARCHAR))) IN ('', 'null', 'none', 'n/a', 'na', 'nan') THEN NULL
                ELSE trim(CAST({pk_col_q} AS VARCHAR))
            END AS v
        FROM {pk_table_q}
    ),
    fk_distincts AS (
        SELECT DISTINCT v
        FROM fk_clean
        WHERE v IS NOT NULL
    ),
    inter AS (
        SELECT COUNT(*) AS intersection_distinct
        FROM fk_distincts f
        INNER JOIN pk_clean p
            ON f.v = p.v
        WHERE p.v IS NOT NULL
    ),
    fk_stats AS (
        SELECT COUNT(*) AS fk_distinct
        FROM fk_distincts
    ),
    pk_stats AS (
        SELECT COUNT(*) AS pk_distinct
        FROM pk_clean
        WHERE v IS NOT NULL
    ),
    row_stats AS (
        SELECT
            COUNT(*) AS fk_non_null_rows,
            SUM(CASE WHEN p.v IS NOT NULL THEN 1 ELSE 0 END) AS matched_rows
        FROM fk_clean f
        LEFT JOIN pk_clean p
            ON f.v = p.v
        WHERE f.v IS NOT NULL
    ),
    fk_num AS (
        SELECT
            MIN(x) AS fk_min,
            MAX(x) AS fk_max
        FROM (
            SELECT TRY_CAST(v AS BIGINT) AS x
            FROM fk_clean
            WHERE v IS NOT NULL
        ) q
        WHERE x IS NOT NULL
    ),
    pk_num AS (
        SELECT
            MIN(x) AS pk_min,
            MAX(x) AS pk_max
        FROM (
            SELECT TRY_CAST(v AS BIGINT) AS x
            FROM pk_clean
            WHERE v IS NOT NULL
        ) q
        WHERE x IS NOT NULL
    )
    SELECT
        fk_stats.fk_distinct,
        pk_stats.pk_distinct,
        row_stats.fk_non_null_rows,
        row_stats.matched_rows,
        inter.intersection_distinct,
        inter.intersection_distinct * 1.0 / NULLIF(fk_stats.fk_distinct, 0) AS distinct_coverage,
        row_stats.matched_rows * 1.0 / NULLIF(row_stats.fk_non_null_rows, 0) AS row_coverage,
        fk_num.fk_min,
        fk_num.fk_max,
        pk_num.pk_min,
        pk_num.pk_max
    FROM fk_stats, pk_stats, row_stats, inter, fk_num, pk_num
    """


def discover_ind_unary_sql(
    *,
    store: DuckDBTableStore,
    ucc_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    min_distinct_child: int = 20,
    min_non_null_rows: int = 20,
    min_distinct_coverage: float = 0.90,
    dtype_families: tuple[str, ...] = ("int", "string"),
    min_fk_unique_ratio_non_null: float = 0.02,
    top1_gate_distinct_max: int = 50,
    max_top1_ratio: float = 0.90,
    min_string_distinct_child: int = 20,
    max_string_len_spread: int = 30,
    max_string_avg_len: float = 40.0,
) -> pd.DataFrame:
    """
    SQL-backed strict unary IND discovery.

    Preserves original logic shape:
    - choose unary UCC parents
    - choose FK-like children from profiles
    - test inclusion per compatible pair
    """
    required_ucc = {"table_name", "column_name", "dtype_family"}
    missing_ucc = required_ucc - set(ucc_df.columns)
    if missing_ucc:
        raise ValueError(f"ucc_df missing required columns: {sorted(missing_ucc)}")

    required_prof = {
        "table_name", "column_name", "dtype_family",
        "null_ratio", "n_rows", "n_unique", "unique_ratio",
        "unique_ratio_non_null", "top1_ratio",
        "avg_len", "min_len", "max_len",
    }
    missing_prof = required_prof - set(profiles_df.columns)
    if missing_prof:
        raise ValueError(f"profiles_df missing required columns: {sorted(missing_prof)}")

    base_cols = [
        "fk_table", "fk_column", "pk_table", "pk_column",
        "fk_dtype_family", "pk_dtype_family",
        "fk_non_null_rows", "fk_distinct",
        "pk_distinct", "intersection_distinct",
        "distinct_coverage", "row_coverage",
        "matched_rows",
        "fk_min", "fk_max", "pk_min", "pk_max",
    ]

    # ------------------------------------------------------------------
    # Parents -> unary UCCs only
    # ------------------------------------------------------------------
    parent = ucc_df.copy()
    parent = parent[parent["dtype_family"].isin(dtype_families)].copy()

    # Optional -> if pruning flags are attached to profiles_df, do not use rejected columns as parents
    if "reject" in profiles_df.columns:
        rejected_pairs = set(
            zip(
                profiles_df.loc[profiles_df["reject"], "table_name"].astype(str),
                profiles_df.loc[profiles_df["reject"], "column_name"].astype(str),
            )
        )
        keep_mask = [
            (str(t), str(c)) not in rejected_pairs
            for t, c in zip(parent["table_name"], parent["column_name"])
        ]
        parent = parent.loc[keep_mask].copy()

    if parent.empty:
        return pd.DataFrame(columns=base_cols)

    parent = parent[["table_name", "column_name", "dtype_family"]].drop_duplicates()
    parent = parent.rename(
        columns={
            "table_name": "pk_table",
            "column_name": "pk_column",
            "dtype_family": "pk_dtype_family",
        }
    )

    # ------------------------------------------------------------------
    # Children -> same pruning rules as your original strict logic
    # ------------------------------------------------------------------
    ucc_keys = set(zip(ucc_df["table_name"].astype(str), ucc_df["column_name"].astype(str)))

    child = profiles_df.copy()

    # Global pruning -> remove obvious junk before strict child selection
    if "reject" in child.columns:
        child = child[~child["reject"]].copy()

    child = child[child["dtype_family"].isin(dtype_families)].copy()

    child_keys = list(zip(child["table_name"].astype(str), child["column_name"].astype(str)))
    child = child[[k not in ucc_keys for k in child_keys]].copy()

    child = child[child["null_ratio"] < 0.95].copy()
    child = child[child["unique_ratio_non_null"] >= float(min_fk_unique_ratio_non_null)].copy()

    low_distinct = child["n_unique"].astype(float) <= float(top1_gate_distinct_max)
    dominated = child["top1_ratio"].astype(float) > float(max_top1_ratio)
    child = child[~(low_distinct & dominated)].copy()

    is_string = child["dtype_family"].astype(str) == "string"
    child = child[~is_string | (child["n_unique"].astype(float) >= float(min_string_distinct_child))].copy()

    len_spread = (child["max_len"].astype(float) - child["min_len"].astype(float)).fillna(0.0)
    avg_len = child["avg_len"].astype(float).fillna(0.0)

    child = child[
        ~is_string | (
            (len_spread <= float(max_string_len_spread)) &
            (avg_len <= float(max_string_avg_len))
        )
    ].copy()

    child = child[["table_name", "column_name", "dtype_family"]].drop_duplicates()
    child = child.rename(
        columns={
            "table_name": "fk_table",
            "column_name": "fk_column",
            "dtype_family": "fk_dtype_family",
        }
    )

    if child.empty:
        return pd.DataFrame(columns=base_cols)

    # ------------------------------------------------------------------
    # Evaluate compatible pairs
    # ------------------------------------------------------------------
    edges: list[dict[str, object]] = []

    for crow in child.itertuples(index=False):
        fk_table = str(crow.fk_table)
        fk_col = str(crow.fk_column)
        fk_dtype = str(crow.fk_dtype_family)

        # get child profile row
        prof_match = profiles_df[
            (profiles_df["table_name"] == fk_table) &
            (profiles_df["column_name"] == fk_col)
        ]
        if prof_match.empty:
            continue

        prof_row = prof_match.iloc[0]
        fk_non_null_profile = int(prof_row["n_rows"] - prof_row["n_rows"] * prof_row["null_ratio"])
        fk_distinct_profile = int(prof_row["n_unique"])

        if fk_non_null_profile < int(min_non_null_rows):
            continue

        min_dist = min_distinct_child if fk_dtype != "string" else min_string_distinct_child
        if fk_distinct_profile < int(min_dist):
            continue

        compatible_parents = parent[
            (parent["pk_dtype_family"] == fk_dtype) &
            (parent["pk_table"] != fk_table)
        ]

        for prow in compatible_parents.itertuples(index=False):
            pk_table = str(prow.pk_table)
            pk_col = str(prow.pk_column)
            pk_dtype = str(prow.pk_dtype_family)

            sql = _build_pair_metrics_sql(fk_table, fk_col, pk_table, pk_col)
            pair_df = store.fetchdf(sql)

            if pair_df.empty:
                continue

            m = pair_df.iloc[0]

            fk_non_null_rows = int(m["fk_non_null_rows"] or 0)
            fk_distinct = int(m["fk_distinct"] or 0)
            pk_distinct = int(m["pk_distinct"] or 0)
            intersection_distinct = int(m["intersection_distinct"] or 0)
            matched_rows = int(m["matched_rows"] or 0)

            distinct_coverage = float(m["distinct_coverage"] or 0.0)
            row_coverage = float(m["row_coverage"] or 0.0)

            if fk_non_null_rows < int(min_non_null_rows):
                continue

            if fk_distinct < int(min_dist):
                continue

            if distinct_coverage < float(min_distinct_coverage):
                continue

            edges.append(
                {
                    "fk_table": fk_table,
                    "fk_column": fk_col,
                    "pk_table": pk_table,
                    "pk_column": pk_col,
                    "fk_dtype_family": fk_dtype,
                    "pk_dtype_family": pk_dtype,
                    "fk_non_null_rows": fk_non_null_rows,
                    "fk_distinct": fk_distinct,
                    "pk_distinct": pk_distinct,
                    "intersection_distinct": intersection_distinct,
                    "distinct_coverage": round(distinct_coverage, 6),
                    "row_coverage": round(row_coverage, 6),
                    "matched_rows": matched_rows,
                    "fk_min": m["fk_min"],
                    "fk_max": m["fk_max"],
                    "pk_min": m["pk_min"],
                    "pk_max": m["pk_max"],
                }
            )

    out = pd.DataFrame(edges)
    if out.empty:
        return pd.DataFrame(columns=base_cols)

    out = out.sort_values(
        by=["distinct_coverage", "row_coverage", "fk_non_null_rows", "fk_distinct"],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    return out