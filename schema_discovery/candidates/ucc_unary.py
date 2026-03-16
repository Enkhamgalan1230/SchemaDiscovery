"""
Unary UCC discovery.

UCC = Unique Column Combination.
Unary: a single column is unique and typically non null.

This module extracts unary UCC candidates from the profiling output.
These candidates are later used as possible referenced keys when searching
for relationships between tables.
"""

from __future__ import annotations

import pandas as pd


def discover_ucc_unary(
    profiles_df: pd.DataFrame,
    unique_ratio_min: float = 1.0,
    null_ratio_max: float = 0.0,
    use_flag_if_present: bool = True,
) -> pd.DataFrame:
    required = {
        "table_name",
        "column_name",
        "dtype_family",
        "n_rows",
        "unique_ratio",
        "null_ratio",
    }
    missing = required - set(profiles_df.columns)
    if missing:
        raise ValueError(f"profiles_df missing required columns: {sorted(missing)}")

    if use_flag_if_present and "is_unary_ucc" in profiles_df.columns:
        mask = profiles_df["is_unary_ucc"].astype(bool)
        if unique_ratio_min != 1.0 or null_ratio_max != 0.0:
            mask = mask | (
                (profiles_df["unique_ratio"] >= unique_ratio_min)
                & (profiles_df["null_ratio"] <= null_ratio_max)
            )
    else:
        mask = (
            (profiles_df["unique_ratio"] >= unique_ratio_min)
            & (profiles_df["null_ratio"] <= null_ratio_max)
        )

    ucc = profiles_df.loc[
        mask,
        ["table_name", "column_name", "dtype_family", "n_rows", "unique_ratio", "null_ratio"],
    ].copy()

    ucc["ucc_type"] = "unary"

    return ucc.sort_values(["table_name", "column_name"], kind="mergesort").reset_index(drop=True)