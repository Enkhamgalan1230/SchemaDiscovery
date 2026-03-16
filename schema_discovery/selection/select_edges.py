"""
Edge selection.

Takes scored FK candidates and selects one best parent per (fk_table, fk_column),
then classifies relationship type (many_to_one vs one_to_one, optional vs required).

Key change (Option A):
- Keep useful evidence columns in the final edges output so the API can return them.
"""

from __future__ import annotations

import pandas as pd


def select_best_edges(
    scored_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    min_score: float = 0.80,
    optional_null_ratio_min: float = 0.01,
    one_to_one_unique_ratio_min: float = 0.999999,
    max_edges_per_fk: int = 3,
) -> pd.DataFrame:
    required_scored = {
        "fk_table", "fk_column", "pk_table", "pk_column",
        "score", "distinct_coverage", "row_coverage",
        "fk_distinct", "fk_non_null_rows",
    }
    missing = required_scored - set(scored_df.columns)
    if missing:
        raise ValueError(f"scored_df missing required columns: {sorted(missing)}")

    required_prof = {"table_name", "column_name", "null_ratio", "unique_ratio"}
    missing_prof = required_prof - set(profiles_df.columns)
    if missing_prof:
        raise ValueError(f"profiles_df missing required columns: {sorted(missing_prof)}")

    # 1) Filter by minimum score
    df = scored_df[scored_df["score"] >= float(min_score)].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "relationship",
            "fk_table", "fk_column", "pk_table", "pk_column",
            "score", "distinct_coverage", "row_coverage",
            "cardinality", "optional",
        ])

    # 2) Pick best parent per FK column
    sort_cols = ["fk_table", "fk_column", "score", "distinct_coverage", "row_coverage"]
    sort_asc = [True, True, False, False, False]
    if "name_sim" in df.columns:
        sort_cols.append("name_sim")
        sort_asc.append(False)

    df = df.sort_values(by=sort_cols, ascending=sort_asc, kind="mergesort")
    df = df.groupby(["fk_table", "fk_column"], as_index=False).head(int(max_edges_per_fk)).reset_index(drop=True)

    # 3) Attach FK stats (optional + cardinality)
    prof_lookup = profiles_df.set_index(["table_name", "column_name"])

    def _get_stat(t: str, c: str, field: str, default: float = 0.0) -> float:
        try:
            return float(prof_lookup.loc[(t, c), field])
        except Exception:
            return float(default)

    optional_flags = []
    cardinalities = []

    for r in df.itertuples(index=False):
        fk_null_ratio = _get_stat(r.fk_table, r.fk_column, "null_ratio", default=0.0)

        # Prefer non-null uniqueness if available
        if "unique_ratio_non_null" in profiles_df.columns:
            fk_unique_ratio = _get_stat(r.fk_table, r.fk_column, "unique_ratio_non_null", default=0.0)
        else:
            fk_unique_ratio = _get_stat(r.fk_table, r.fk_column, "unique_ratio", default=0.0)

        optional_flags.append(fk_null_ratio >= float(optional_null_ratio_min))
        cardinalities.append(
            "one_to_one" if fk_unique_ratio >= float(one_to_one_unique_ratio_min) else "many_to_one"
        )

    df["optional"] = optional_flags
    df["cardinality"] = cardinalities

    # 4) Display string
    df["relationship"] = df["fk_table"] + "." + df["fk_column"] + " -> " + df["pk_table"] + "." + df["pk_column"]

    # 5) Keep base fields + preserve useful evidence if present
    base_keep = [
        "relationship",
        "fk_table", "fk_column",
        "pk_table", "pk_column",
        "score", "distinct_coverage", "row_coverage",
        "cardinality", "optional",
    ]

    extra_if_present = [
        # evidence
        "fk_non_null_rows", "fk_distinct", "pk_distinct", "intersection_distinct", "matched_rows",
        # scoring extras
        "name_sim", "range_penalty",
        # min/max if you ever want to expose it later
        "fk_min", "fk_max", "pk_min", "pk_max",
    ]

    keep = base_keep + [c for c in extra_if_present if c in df.columns]

    return (
        df[keep]
        .sort_values(by=["score", "distinct_coverage", "row_coverage"], ascending=[False, False, False], kind="mergesort")
        .reset_index(drop=True)
    )

