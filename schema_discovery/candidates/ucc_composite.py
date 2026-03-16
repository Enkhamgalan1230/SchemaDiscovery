from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from schema_discovery.quality.key_normalisation import normalise_null_like


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class CompositeUCCConfig:
    # search limits
    max_k: int = 3                 # 2 or 3 is usually enough
    max_cols_per_table: int = 12   # cap to control combinations

    # column prefilter gates
    null_ratio_max: float = 0.01   # used only when require_no_nulls_in_key=False
    min_distinct: int = 10
    max_avg_len_string: int = 40

    # strict PK-like key requirement
    require_no_nulls_in_key: bool = True

    # practical guardrails
    exclude_float_cols: bool = True

    # optional: block obvious measure columns by name (helps avoid noise keys)
    block_measure_names: bool = True


# -----------------------------
# Row-preserving normalisation
# -----------------------------
def norm_key_series_keep_rows(
    s: pd.Series,
    *,
    normalise_string_nulls: bool = True,
) -> pd.Series:
    """
    Normalise values for key comparison WITHOUT dropping rows.

    This is critical for composite key uniqueness testing because dropping rows
    breaks row alignment and can create false uniqueness.
    """
    out = s.copy()

    if normalise_string_nulls:
        out = normalise_null_like(out)

    # numeric
    if pd.api.types.is_numeric_dtype(out):
        x = pd.to_numeric(out, errors="coerce")  # keeps NaN positions
        # cast integer-like numeric to Int64 (nullable)
        arr = x.to_numpy()
        mask = pd.notna(arr)
        if mask.any():
            try:
                if np.isclose(arr[mask], np.round(arr[mask])).all():
                    return x.round().astype("Int64").astype(object)
            except Exception:
                pass
        return x.astype(object)

    # string / object
    return out.astype("string").str.strip().astype(object)


def _looks_measure_like(col: str) -> bool:
    s = str(col).strip().lower()
    tokens = [
        "minute", "time", "second", "day", "month", "year", "age",
        "status", "stage", "round", "week",
        "amount", "price", "total", "qty", "quantity", "count", "score", "rating",
    ]
    return any(t in s for t in tokens)


# -----------------------------
# Candidate selection
# -----------------------------
def _candidate_columns_for_table(
    table: str,
    profiles_df: pd.DataFrame,
    edges_df: pd.DataFrame | None,
    cfg: CompositeUCCConfig,
) -> list[str]:
    table = str(table)

    if profiles_df is None or profiles_df.empty:
        return []
    if not {"table_name", "column_name"}.issubset(profiles_df.columns):
        return []

    tprof = profiles_df[profiles_df["table_name"].astype(str) == table].copy()
    if tprof.empty:
        return []

    # 1) remove unary ucc columns (minimality prune)
    if "is_unary_ucc" in tprof.columns:
        tprof = tprof[~tprof["is_unary_ucc"].fillna(False).astype(bool)].copy()
    else:
        if {"unique_ratio", "null_ratio"}.issubset(tprof.columns):
            tprof = tprof[~((tprof["unique_ratio"] >= 1.0) & (tprof["null_ratio"] == 0.0))].copy()

    if tprof.empty:
        return []

    # 2) null gate
    if "null_ratio" in tprof.columns:
        if cfg.require_no_nulls_in_key:
            # if key must have no nulls -> every key component must have no nulls
            tprof = tprof[tprof["null_ratio"].astype(float) == 0.0].copy()
        else:
            tprof = tprof[tprof["null_ratio"].astype(float) <= float(cfg.null_ratio_max)].copy()

    # 3) avoid tiny domains
    if "n_unique" in tprof.columns:
        tprof = tprof[tprof["n_unique"].astype(float) >= float(cfg.min_distinct)].copy()

    # 4) avoid long free text
    if {"dtype_family", "avg_len"}.issubset(tprof.columns):
        mask_text = (tprof["dtype_family"].astype(str) == "string") & (
            tprof["avg_len"].astype(float).fillna(0.0) > float(cfg.max_avg_len_string)
        )
        tprof = tprof[~mask_text].copy()

    # 5) exclude float columns
    if cfg.exclude_float_cols and "dtype_family" in tprof.columns:
        tprof = tprof[tprof["dtype_family"].astype(str) != "float"].copy()

    # 6) optional: drop obvious measure-like columns by name
    if cfg.block_measure_names:
        tprof = tprof[~tprof["column_name"].astype(str).map(_looks_measure_like)].copy()

    if tprof.empty:
        return []

    cols = tprof["column_name"].astype(str).tolist()

    # 7) boost columns that already look like FKs (edges evidence)
    boost: set[str] = set()
    if edges_df is not None and (not edges_df.empty) and {"fk_table", "fk_column"}.issubset(edges_df.columns):
        boost = set(
            edges_df.loc[edges_df["fk_table"].astype(str) == table, "fk_column"]
            .astype(str)
            .tolist()
        )

    boosted = [c for c in cols if c in boost]
    rest = [c for c in cols if c not in boost]
    out = boosted + rest

    return out[: int(cfg.max_cols_per_table)]


# -----------------------------
# Uniqueness testing
# -----------------------------
def _is_unique_combo(
    df: pd.DataFrame,
    cols: list[str],
    require_no_nulls: bool,
) -> tuple[bool, dict[str, Any]]:
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return False, {"n_rows_used": 0, "n_dup_rows": 0, "n_null_rows": 0}

    sub = df[cols].copy()

    # IMPORTANT: row-preserving normalisation
    for c in cols:
        sub[c] = norm_key_series_keep_rows(sub[c])

    null_mask = sub.isna().any(axis=1)
    n_null_rows = int(null_mask.sum())

    if require_no_nulls and n_null_rows > 0:
        return False, {"n_rows_used": int(len(sub)), "n_dup_rows": 0, "n_null_rows": n_null_rows}

    if not require_no_nulls:
        sub = sub[~null_mask].copy()

    n_rows_used = int(len(sub))
    if n_rows_used == 0:
        return False, {"n_rows_used": 0, "n_dup_rows": 0, "n_null_rows": n_null_rows}

    # duplicated checks uniqueness across all rows used
    dup_mask = sub.duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())

    return (n_dup_rows == 0), {"n_rows_used": n_rows_used, "n_dup_rows": n_dup_rows, "n_null_rows": n_null_rows}


# -----------------------------
# Main API
# -----------------------------
def discover_ucc_composite(
    dfs: dict[str, pd.DataFrame],
    profiles_df: pd.DataFrame,
    ucc_unary_df: pd.DataFrame,
    edges_df: pd.DataFrame | None = None,
    cfg: CompositeUCCConfig | None = None,
) -> pd.DataFrame:
    """
    Returns composite UCC candidates (minimal) up to cfg.max_k.

    Output schema:
      table_name, columns, k, ucc_type, n_rows_used
    """
    if cfg is None:
        cfg = CompositeUCCConfig()

    if profiles_df is None or profiles_df.empty:
        return pd.DataFrame(columns=["table_name", "columns", "k", "ucc_type", "n_rows_used"])

    # unary ucc lookup for minimality prune
    unary_map: dict[str, set[str]] = {}
    if (
        ucc_unary_df is not None
        and (not ucc_unary_df.empty)
        and {"table_name", "column_name"}.issubset(ucc_unary_df.columns)
    ):
        for t, c in ucc_unary_df[["table_name", "column_name"]].itertuples(index=False):
            unary_map.setdefault(str(t), set()).add(str(c))

    out_rows: list[dict[str, Any]] = []

    for table, df in dfs.items():
        table = str(table)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        cand_cols = _candidate_columns_for_table(table, profiles_df, edges_df, cfg)
        if len(cand_cols) < 2:
            continue

        found: list[tuple[str, ...]] = []

        for k in range(2, int(cfg.max_k) + 1):
            for comb in combinations(cand_cols, k):
                comb = tuple(map(str, comb))

                # minimality prune 1: if any unary ucc is inside comb, skip
                if table in unary_map and any(c in unary_map[table] for c in comb):
                    continue

                # minimality prune 2: skip supersets of already found composite keys
                comb_set = set(comb)
                if any(set(prev).issubset(comb_set) for prev in found):
                    continue

                ok, st = _is_unique_combo(
                    df=df,
                    cols=list(comb),
                    require_no_nulls=bool(cfg.require_no_nulls_in_key),
                )
                if ok:
                    found.append(comb)
                    out_rows.append(
                        {
                            "table_name": table,
                            "columns": list(comb),
                            "k": k,
                            "ucc_type": "composite",
                            "n_rows_used": int(st["n_rows_used"]),
                        }
                    )

    if not out_rows:
        return pd.DataFrame(columns=["table_name", "columns", "k", "ucc_type", "n_rows_used"]).reset_index(drop=True)

    return (
        pd.DataFrame(out_rows)
        .sort_values(["table_name", "k"], kind="mergesort")
        .reset_index(drop=True)
    )
