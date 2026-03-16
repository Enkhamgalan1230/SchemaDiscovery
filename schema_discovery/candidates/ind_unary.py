"""
Unary IND discovery.

IND = Inclusion Dependency: values in column A are included in column B (A ⊆ B).
Unary means one column on each side.

This module generates candidate cross-table links by testing child columns
against candidate unique columns (UCCs) from other tables.

Output is not just True or False. It returns coverage metrics so later stages
can score and select the most plausible relationships.

Recent changes:

Parent key domains are computed once and cached. This is the big speedup.
Removed per row apply on profiles_df for excluding UCCs.
Added matched_rows because it helps interpret row coverage.
Output schema is consistent even when empty.
"""

from __future__ import annotations

import pandas as pd
import numpy as np 

from schema_discovery.quality.key_normalisation import norm_key_series


def discover_ind_unary(
    dfs: dict[str, pd.DataFrame],
    ucc_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    min_distinct_child: int = 20,
    min_non_null_rows: int = 20,
    min_distinct_coverage: float = 0.90,
    dtype_families: tuple[str, ...] = ("int","string"),
    min_fk_unique_ratio_non_null: float = 0.02,
    top1_gate_distinct_max: int = 50,
    max_top1_ratio: float = 0.90,
    min_string_distinct_child: int = 20,
    max_string_len_spread: int = 30,
    max_string_avg_len: float = 40.0,
    max_pk_distinct_cache: int = 2_000_000,
) -> pd.DataFrame:
    
    required_ucc = {"table_name", "column_name", "dtype_family"}
    missing_ucc = required_ucc - set(ucc_df.columns)
    if missing_ucc:
        raise ValueError(f"ucc_df missing required columns: {sorted(missing_ucc)}")

    required_prof = {
        "table_name", "column_name", "dtype_family",
        "null_ratio", "n_rows", "n_unique", "unique_ratio",
        "unique_ratio_non_null",
        "top1_ratio",
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

    # -----------------------------
    # 1) Build parent candidates list (UCCs) and precompute their domains
    # -----------------------------
    parent = ucc_df.copy()
    parent = parent[parent["dtype_family"].isin(dtype_families)].copy()

    parent_rows = []
    for pk_table, pk_col, pk_dtype in parent[["table_name", "column_name", "dtype_family"]].itertuples(index=False):
        if pk_table not in dfs or pk_col not in dfs[pk_table].columns:
            continue
        parent_rows.append((pk_table, pk_col, pk_dtype))
    parent = pd.DataFrame(parent_rows, columns=["pk_table", "pk_column", "pk_dtype_family"])

    if parent.empty:
        return pd.DataFrame(columns=base_cols)

    def _minmax_if_int(s: pd.Series, dtype_family: str):
        if dtype_family != "int" or s.empty:
            return None, None
        x = pd.to_numeric(pd.Series(s, dtype="object"), errors="coerce").dropna()
        if x.empty:
            return None, None
        return int(x.min()), int(x.max())
    
    def _hash_u64(s: pd.Series) -> pd.Series:
        """
        Return stable uint64 hashes for values in s (object/string/int all supported).
        Uses pandas hashing (fast, vectorised).
        """
        return pd.util.hash_pandas_object(pd.Series(s, dtype="object"), index=False).astype("uint64")


    # Precompute parent sets once: pk_set, pk_distinct, min, max
    parent_cache: dict[tuple[str, str], dict[str, object]] = {}
    for pk_table, pk_col, pk_dtype in parent.itertuples(index=False):
        s_pk = norm_key_series(dfs[pk_table][pk_col])
        if s_pk.empty:
            continue

        pk_hash = _hash_u64(s_pk)
        pk_hash_distinct = pk_hash.drop_duplicates()

        pk_hset = set(pk_hash_distinct.tolist())
        if not pk_hset:
            continue

        # optional safety cap
        if len(pk_hset) > int(max_pk_distinct_cache):
            continue

        pk_min, pk_max = _minmax_if_int(s_pk, pk_dtype)

        parent_cache[(pk_table, pk_col)] = {
            "pk_dtype_family": pk_dtype,
            "pk_hset": pk_hset,
            "pk_distinct": int(len(pk_hset)),
            "pk_min": pk_min,
            "pk_max": pk_max,
        }


    if not parent_cache:
        return pd.DataFrame(columns=base_cols)

    # -----------------------------
    # 2) Choose child columns to test (non-UCC columns, compatible dtype)
    # -----------------------------
    # Build a fast lookup of UCC columns so we do not test them as children
    ucc_keys = set(zip(ucc_df["table_name"].astype(str), ucc_df["column_name"].astype(str)))

    child = profiles_df.copy()
    child = child[child["dtype_family"].isin(dtype_families)].copy()

    # vectorised filter: exclude UCC columns
    child_keys = list(zip(child["table_name"].astype(str), child["column_name"].astype(str)))
    child = child[[k not in ucc_keys for k in child_keys]].copy()

    # cheap pruning: skip extremely sparse columns
    child = child[child["null_ratio"] < 0.95].copy()

    # ---------------------------------
    # NEW: FK-likeness gates
    # ---------------------------------

    # Gate 1: minimum non-null uniqueness among non-null rows
    child = child[child["unique_ratio_non_null"] >= float(min_fk_unique_ratio_non_null)].copy()

    # Gate 2: top1 dominance gate, only applied to low-distinct columns
    # If a column has few distinct values and one value dominates, it is almost never an FK.
    low_distinct = child["n_unique"].astype(float) <= float(top1_gate_distinct_max)
    dominated = child["top1_ratio"].astype(float) > float(max_top1_ratio)
    child = child[~(low_distinct & dominated)].copy()

    is_string = child["dtype_family"].astype(str) == "string"

    # enough distinct values for string FKs
    child = child[~is_string | (child["n_unique"].astype(float) >= float(min_string_distinct_child))].copy()

    # block free-text by length behaviour
    len_spread = (child["max_len"].astype(float) - child["min_len"].astype(float)).fillna(0.0)
    avg_len = child["avg_len"].astype(float).fillna(0.0)

    child = child[
        ~is_string
        | (
            (len_spread <= float(max_string_len_spread))
            & (avg_len <= float(max_string_avg_len))
        )
    ].copy()

    # only keep columns that exist in dfs
    child_rows = []
    for fk_table, fk_col, fk_dtype in child[["table_name", "column_name", "dtype_family"]].itertuples(index=False):
        if fk_table not in dfs or fk_col not in dfs[fk_table].columns:
            continue
        child_rows.append((fk_table, fk_col, fk_dtype))
    child = pd.DataFrame(child_rows, columns=["fk_table", "fk_column", "fk_dtype_family"])

    if child.empty:
        return pd.DataFrame(columns=base_cols)

    # -----------------------------
    # 3) Test inclusion (distinct coverage + row coverage)
    # -----------------------------
    edges = []

    for fk_table, fk_col, fk_dtype in child.itertuples(index=False):
        s_fk = norm_key_series(dfs[fk_table][fk_col])
        fk_non_null = int(len(s_fk))
        if fk_non_null < min_non_null_rows:
            continue

        fk_hash = _hash_u64(s_fk)

        fk_hash_distinct = fk_hash.drop_duplicates()
        fk_distinct = int(len(fk_hash_distinct))

        min_dist = min_distinct_child if fk_dtype != "string" else min_string_distinct_child
        if fk_distinct < int(min_dist):
            continue

        fk_hset = set(fk_hash_distinct.tolist())
        if not fk_hset:
            continue


        fk_min, fk_max = _minmax_if_int(s_fk, fk_dtype)

        for (pk_table, pk_col), pk_info in parent_cache.items():
            if pk_table == fk_table:
                continue

            pk_dtype = pk_info["pk_dtype_family"]
            if pk_dtype != fk_dtype:
                continue

            pk_hset = pk_info["pk_hset"]
            pk_distinct = int(pk_info["pk_distinct"])

            inter = fk_hset.intersection(pk_hset)
            intersection_distinct = int(len(inter))
            distinct_coverage = intersection_distinct / fk_distinct if fk_distinct else 0.0
            if distinct_coverage < min_distinct_coverage:
                continue

            matched_rows = int(fk_hash.isin(pk_hset).sum())
            row_coverage = matched_rows / fk_non_null if fk_non_null else 0.0

            edges.append(
                {
                    "fk_table": fk_table,
                    "fk_column": fk_col,
                    "pk_table": pk_table,
                    "pk_column": pk_col,
                    "fk_dtype_family": fk_dtype,
                    "pk_dtype_family": pk_dtype,
                    "fk_non_null_rows": fk_non_null,
                    "fk_distinct": fk_distinct,
                    "pk_distinct": pk_distinct,
                    "fk_min": fk_min,
                    "fk_max": fk_max,
                    "pk_min": pk_info["pk_min"],
                    "pk_max": pk_info["pk_max"],
                    "intersection_distinct": intersection_distinct,
                    "distinct_coverage": round(distinct_coverage, 6),
                    "row_coverage": round(row_coverage, 6),
                    "matched_rows": matched_rows,
                }
            )

    out = pd.DataFrame(edges)
    if out.empty:
        return pd.DataFrame(columns=base_cols)

    # Sort best evidence first for readability
    out = out.sort_values(
        by=["distinct_coverage", "row_coverage", "fk_non_null_rows", "fk_distinct"],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    return out
