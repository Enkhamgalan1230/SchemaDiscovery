# schema_discovery/quality/duplicates.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Iterable
import pandas as pd


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
@dataclass(frozen=True)
class DuplicateChecksConfig:
    """
    Duplicate + overlap diagnostics config.

    You have 3 "levels" of duplication:
    1) Cell-level duplicates -> repeated values inside a column (only for key or natural-key columns)
    2) Relational duplicates -> repeated rows (exact, or repeated on a subset of columns)
    3) Identifier conflicts -> same entity represented by multiple IDs or mappings

    Defaults aim to avoid noise:
    - ID-like columns are treated as keys by default
    - Natural keys (email, phone, etc) are checked separately by name patterns
    """

    # Basic profiler thresholds
    key_min_non_null_rows: int = 20
    key_min_unique_ratio: float = 0.98

    # Only treat ID-like columns as "key candidates"
    key_id_like_only: bool = True

    # Natural key patterns (cell-level duplicates) even if not ID-like
    natural_key_name_substrings: tuple[str, ...] = (
        "email",
        "e_mail",
        "phone",
        "mobile",
        "msisdn",
        "account",
        "acct",
        "username",
        "user_name",
    )
    # If a natural key column has very low uniqueness, it is usually not a key -> skip to reduce noise
    natural_key_min_unique_ratio: float = 0.70

    # Cross-table overlap thresholds
    overlap_min_ratio_smaller_set: float = 0.10
    max_examples: int = 10

    # Signature mapping multiple IDs
    signature_unique_ratio_max: float = 0.95
    signature_min_cols: int = 2

    signature_exclude_low_cardinality: bool = True
    signature_low_cardinality_max_distinct: int = 5

    # For strong natural keys, allow signature with 1 column (example: email)
    allow_single_col_signature_for_natural_keys: bool = True

    # Relational duplicates
    relational_duplicate_subset_keys: dict[str, list[list[str]]] = field(
        default_factory=dict
    )
    # Example:
    # relational_duplicate_subset_keys = {
    #   "payments": [["order_id", "amount", "status"], ["payment_id"]],
    #   "customers": [["email"]],
    # }

    # Relationship table checks
    # You can define expected mapping rules for tables that look like link tables
    # Example:
    # relationship_tables = {
    #   "customer_identity_map": {"left": "customer_id", "right": "crm_customer_id", "expected": "one_to_one"}
    # }
    relationship_tables: dict[str, dict[str, str]] = field(default_factory=dict)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _norm_colname(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _is_id_like(col: str) -> bool:
    c = _norm_colname(col)
    return (
        c.endswith("_id")
        or c == "id"
        or "uuid" in c
        or c.startswith("pk_")
        or c.startswith("fk_")
    )


def _is_natural_key_like(col: str, cfg: DuplicateChecksConfig) -> bool:
    c = _norm_colname(col)
    return any(pat in c for pat in cfg.natural_key_name_substrings)


def _profile_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_rows = int(len(df))

    for col in df.columns:
        s = df[col]
        n_null = int(s.isna().sum())
        n_non_null = int(n_rows - n_null)
        n_unique_non_null = int(s.dropna().nunique())
        unique_ratio_non_null = (n_unique_non_null / n_non_null) if n_non_null else 0.0

        rows.append(
            {
                "column": str(col),
                "n_rows": n_rows,
                "n_null": n_null,
                "n_non_null": n_non_null,
                "n_unique_non_null": n_unique_non_null,
                "unique_ratio_non_null": float(unique_ratio_non_null),
                "dtype": str(s.dtype),
            }
        )

    return pd.DataFrame(rows)


def _pick_key_candidates(profile_df: pd.DataFrame, cfg: DuplicateChecksConfig) -> list[str]:
    """
    Key candidate selection:
    - By default, only ID-like columns qualify as "keys"
    - Unique ratio threshold is used to avoid treating repeating FKs as keys
    """
    if profile_df is None or profile_df.empty:
        return []

    sub = profile_df[
        (profile_df["n_non_null"] >= cfg.key_min_non_null_rows)
        & (profile_df["unique_ratio_non_null"] >= cfg.key_min_unique_ratio)
    ].copy()

    cols = sub["column"].tolist()
    id_like = [c for c in cols if _is_id_like(c)]

    if cfg.key_id_like_only:
        return id_like

    others = [c for c in cols if c not in id_like]
    return id_like + others


def _pick_natural_key_candidates(profile_df: pd.DataFrame, cfg: DuplicateChecksConfig) -> list[str]:
    """
    Natural key candidates are based mainly on name patterns (email, phone, etc).
    We still gate by minimum uniqueness to avoid spammy columns.
    """
    if profile_df is None or profile_df.empty:
        return []

    sub = profile_df[
        (profile_df["n_non_null"] >= cfg.key_min_non_null_rows)
        & (profile_df["unique_ratio_non_null"] >= cfg.natural_key_min_unique_ratio)
        & (profile_df["column"].apply(lambda x: _is_natural_key_like(str(x), cfg)))
    ].copy()

    return sub["column"].tolist()


def _values_set_normalised(s: pd.Series) -> set[str]:
    if s is None:
        return set()

    x = s.dropna()

    # Numeric -> normalize integer-like floats to ints
    if pd.api.types.is_numeric_dtype(x):
        x = pd.to_numeric(x, errors="coerce").dropna()
        if len(x) == 0:
            return set()

        if (x % 1 == 0).all():
            x = x.astype("Int64").astype(str)
        else:
            x = x.astype(float).astype(str)

        return set(x.tolist())

    # Strings or mixed
    x = x.astype(str).str.strip()
    return set(x.tolist())


def _top_duplicate_values(s: pd.Series, max_examples: int) -> list[str]:
    if s is None:
        return []
    vc = s.dropna().astype(str).value_counts()
    dup = vc[vc > 1]
    return dup.head(int(max_examples)).index.tolist()


def _safe_cols(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


# ---------------------------------------------------------
# A1) Cell-level duplicates in key-like columns (within table)
# ---------------------------------------------------------
def find_duplicate_values_in_key_columns(
    df: pd.DataFrame,
    table_name: str,
    cfg: Optional[DuplicateChecksConfig] = None,
    *,
    key_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Flags duplicate values inside columns expected to behave like unique IDs (PK-like).
    """
    cfg = cfg or DuplicateChecksConfig()
    prof = _profile_table(df)

    if key_cols is None:
        key_cols = _pick_key_candidates(prof, cfg)
    key_cols = _safe_cols(df, key_cols or [])

    rows: list[dict[str, Any]] = []
    n_rows_total = int(len(df))

    for col in key_cols:
        s = df[col]
        n_non_null = int(s.notna().sum())
        dup_rows = int(s.dropna().astype(str).duplicated().sum())

        if dup_rows <= 0:
            continue

        rows.append(
            {
                "table": table_name,
                "column": col,
                "issue": "duplicate_values_in_key_column",
                "total_rows": n_rows_total,
                "non_null_rows": n_non_null,
                "rows_duplicated_after_first": dup_rows,
                "duplicate_ratio_non_null": round(float(dup_rows / n_non_null), 6) if n_non_null else 0.0,
                "example_duplicate_values": ", ".join(_top_duplicate_values(s, cfg.max_examples)),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# A2) Cell-level duplicates in natural-key columns (email, phone, etc)
# ---------------------------------------------------------
def find_duplicate_values_in_natural_key_columns(
    df: pd.DataFrame,
    table_name: str,
    cfg: Optional[DuplicateChecksConfig] = None,
    *,
    natural_key_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Same as A1, but focused on natural keys (email, phone, username).
    These are often "unique in real world" even if not named *_id.
    """
    cfg = cfg or DuplicateChecksConfig()
    prof = _profile_table(df)

    if natural_key_cols is None:
        natural_key_cols = _pick_natural_key_candidates(prof, cfg)
    natural_key_cols = _safe_cols(df, natural_key_cols or [])

    rows: list[dict[str, Any]] = []
    n_rows_total = int(len(df))

    for col in natural_key_cols:
        s = df[col]
        n_non_null = int(s.notna().sum())
        dup_rows = int(s.dropna().astype(str).duplicated().sum())

        if dup_rows <= 0:
            continue

        rows.append(
            {
                "table": table_name,
                "column": col,
                "issue": "duplicate_values_in_natural_key_column",
                "total_rows": n_rows_total,
                "non_null_rows": n_non_null,
                "rows_duplicated_after_first": dup_rows,
                "duplicate_ratio_non_null": round(float(dup_rows / n_non_null), 6) if n_non_null else 0.0,
                "example_duplicate_values": ", ".join(_top_duplicate_values(s, cfg.max_examples)),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# B1) Cross-table overlap for same column names
# ---------------------------------------------------------
def find_cross_table_value_overlap_same_name(
    dfs: dict[str, pd.DataFrame],
    cfg: Optional[DuplicateChecksConfig] = None,
) -> pd.DataFrame:
    """
    For any normalized column name that appears in >= 2 tables,
    report overlap of distinct values between each table pair.
    """
    cfg = cfg or DuplicateChecksConfig()

    col_index: dict[str, list[tuple[str, str]]] = {}
    for t, df in dfs.items():
        for c in df.columns:
            col_index.setdefault(_norm_colname(c), []).append((t, str(c)))

    rows: list[dict[str, Any]] = []

    for norm_name, appearances in col_index.items():
        if len(appearances) < 2:
            continue

        for i in range(len(appearances)):
            t1, c1 = appearances[i]
            s1 = _values_set_normalised(dfs[t1][c1])

            for j in range(i + 1, len(appearances)):
                t2, c2 = appearances[j]
                s2 = _values_set_normalised(dfs[t2][c2])

                if not s1 or not s2:
                    continue

                inter = s1.intersection(s2)
                inter_n = len(inter)
                denom = min(len(s1), len(s2))
                overlap_ratio = (inter_n / denom) if denom else 0.0

                if overlap_ratio < cfg.overlap_min_ratio_smaller_set:
                    continue

                examples = list(sorted(list(inter)))[: int(cfg.max_examples)]

                rows.append(
                    {
                        "mode": "same_column_name",
                        "column_name": norm_name,
                        "table_a": t1,
                        "col_a": c1,
                        "distinct_a": len(s1),
                        "table_b": t2,
                        "col_b": c2,
                        "distinct_b": len(s2),
                        "intersection_distinct": inter_n,
                        "overlap_ratio_smaller_set": round(float(overlap_ratio), 6),
                        "example_values": ", ".join(examples),
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# B2) Cross-table overlap for ID-like and natural-key columns even if names differ
# ---------------------------------------------------------
def find_cross_table_value_overlap_id_and_natural(
    dfs: dict[str, pd.DataFrame],
    cfg: Optional[DuplicateChecksConfig] = None,
) -> pd.DataFrame:
    """
    Compares candidate identifier columns across tables even if column names differ.
    This catches things like:
    - customers.customer_id overlapping crm_customers.crm_customer_id (if numbering overlaps)
    - customers.email overlapping crm_customers.email
    """
    cfg = cfg or DuplicateChecksConfig()

    rows: list[dict[str, Any]] = []

    # collect candidates per table
    table_candidates: dict[str, list[str]] = {}
    for t, df in dfs.items():
        prof = _profile_table(df)
        key_cols = _pick_key_candidates(prof, cfg)
        natural_cols = _pick_natural_key_candidates(prof, cfg)
        cand = list(dict.fromkeys(_safe_cols(df, (key_cols or []) + (natural_cols or []))))
        table_candidates[t] = cand

    tables = list(dfs.keys())

    for i in range(len(tables)):
        t1 = tables[i]
        for j in range(i + 1, len(tables)):
            t2 = tables[j]

            for c1 in table_candidates.get(t1, []):
                s1 = _values_set_normalised(dfs[t1][c1])
                if not s1:
                    continue

                for c2 in table_candidates.get(t2, []):
                    s2 = _values_set_normalised(dfs[t2][c2])
                    if not s2:
                        continue

                    inter = s1.intersection(s2)
                    inter_n = len(inter)
                    denom = min(len(s1), len(s2))
                    overlap_ratio = (inter_n / denom) if denom else 0.0

                    if overlap_ratio < cfg.overlap_min_ratio_smaller_set:
                        continue

                    examples = list(sorted(list(inter)))[: int(cfg.max_examples)]

                    rows.append(
                        {
                            "mode": "id_or_natural_key_any_name",
                            "table_a": t1,
                            "col_a": c1,
                            "distinct_a": len(s1),
                            "table_b": t2,
                            "col_b": c2,
                            "distinct_b": len(s2),
                            "intersection_distinct": inter_n,
                            "overlap_ratio_smaller_set": round(float(overlap_ratio), 6),
                            "example_values": ", ".join(examples),
                        }
                    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# C) Values reused across multiple identifier columns in same table
# ---------------------------------------------------------
def find_within_table_id_value_reuse(
    df: pd.DataFrame,
    table_name: str,
    cfg: Optional[DuplicateChecksConfig] = None,
    *,
    id_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Checks if the same identifier values appear in multiple ID-like columns
    inside the same table.
    """
    cfg = cfg or DuplicateChecksConfig()
    prof = _profile_table(df)

    if id_cols is None:
        id_cols = prof[
            (prof["column"].apply(_is_id_like))
            & (prof["n_non_null"] >= cfg.key_min_non_null_rows)
        ]["column"].tolist()

    id_cols = _safe_cols(df, id_cols or [])

    rows: list[dict[str, Any]] = []

    for i in range(len(id_cols)):
        c1 = id_cols[i]
        s1 = _values_set_normalised(df[c1])
        if not s1:
            continue

        for j in range(i + 1, len(id_cols)):
            c2 = id_cols[j]
            s2 = _values_set_normalised(df[c2])
            if not s2:
                continue

            inter = s1.intersection(s2)
            inter_n = len(inter)
            denom = min(len(s1), len(s2))
            overlap_ratio = (inter_n / denom) if denom else 0.0

            if overlap_ratio < cfg.overlap_min_ratio_smaller_set:
                continue

            examples = list(sorted(list(inter)))[: int(cfg.max_examples)]

            rows.append(
                {
                    "table": table_name,
                    "issue": "id_value_reuse_across_id_columns",
                    "id_col_a": c1,
                    "id_col_b": c2,
                    "intersection_distinct": inter_n,
                    "overlap_ratio_smaller_set": round(float(overlap_ratio), 6),
                    "example_values": ", ".join(examples),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# D) Same natural key signature maps to multiple IDs
# ---------------------------------------------------------
def find_same_natural_key_multiple_ids(
    df: pd.DataFrame,
    table_name: str,
    cfg: Optional[DuplicateChecksConfig] = None,
    *,
    id_cols: Optional[list[str]] = None,
    signature_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Detect cases where multiple different IDs are associated with the same signature.
    """
    cfg = cfg or DuplicateChecksConfig()
    prof = _profile_table(df)

    if id_cols is None:
        id_cols = prof[
            (prof["column"].apply(_is_id_like))
            & (prof["n_non_null"] >= cfg.key_min_non_null_rows)
        ]["column"].tolist()

    id_cols = _safe_cols(df, id_cols or [])
    if not id_cols:
        return pd.DataFrame()

    # Signature columns
    if signature_cols is None:
        # Prefer natural keys if present
        natural_cols = _pick_natural_key_candidates(prof, cfg)
        natural_cols = _safe_cols(df, natural_cols or [])

        if cfg.allow_single_col_signature_for_natural_keys and len(natural_cols) >= 1:
            signature_cols = [natural_cols[0]]
        else:
            sig_prof = prof[
                (~prof["column"].apply(_is_id_like))
                & (prof["n_non_null"] >= cfg.key_min_non_null_rows)
                & (prof["unique_ratio_non_null"] <= cfg.signature_unique_ratio_max)
            ].copy()

            if cfg.signature_exclude_low_cardinality:
                sig_prof = sig_prof[sig_prof["n_unique_non_null"] > cfg.signature_low_cardinality_max_distinct]

            signature_cols = sig_prof["column"].tolist()

    signature_cols = _safe_cols(df, signature_cols or [])

    # Validate signature width
    if cfg.allow_single_col_signature_for_natural_keys:
        min_cols = 1 if (len(signature_cols) == 1 and _is_natural_key_like(signature_cols[0], cfg)) else cfg.signature_min_cols
    else:
        min_cols = cfg.signature_min_cols

    if len(signature_cols) < min_cols:
        return pd.DataFrame()

    # Build signature string
    sig_df = df[signature_cols].copy()
    for c in signature_cols:
        sig_df[c] = sig_df[c].fillna("").astype(str).str.strip()
    signature = sig_df.agg("|".join, axis=1)

    rows: list[dict[str, Any]] = []

    for id_col in id_cols:
        tmp = pd.DataFrame({"signature": signature, "id": df[id_col]})
        tmp = tmp.dropna(subset=["id"]).copy()
        tmp["id"] = tmp["id"].astype(str)

        g = tmp.groupby("signature")["id"].nunique()
        bad = g[g > 1]
        if bad.empty:
            continue

        for sig_val, n_ids in bad.head(cfg.max_examples).items():
            subset = df.loc[signature == sig_val, [id_col] + signature_cols].head(cfg.max_examples)

            rows.append(
                {
                    "table": table_name,
                    "issue": "same_signature_multiple_ids",
                    "id_column": id_col,
                    "signature_cols": ", ".join(signature_cols),
                    "signature_example": str(sig_val)[:120],
                    "distinct_ids_for_signature": int(n_ids),
                    "example_rows": subset.to_dict(orient="records"),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# E1) Relational duplicates: exact duplicate rows
# ---------------------------------------------------------
def find_exact_duplicate_rows(
    df: pd.DataFrame,
    table_name: str,
    cfg: Optional[DuplicateChecksConfig] = None,
) -> pd.DataFrame:
    """
    Finds exact duplicate rows (same values across all columns).
    """
    cfg = cfg or DuplicateChecksConfig()

    if df is None or df.empty:
        return pd.DataFrame()

    dup_mask = df.duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    if n_dup == 0:
        return pd.DataFrame()

    examples = df.loc[dup_mask].head(cfg.max_examples).to_dict(orient="records")

    return pd.DataFrame(
        [
            {
                "table": table_name,
                "issue": "exact_duplicate_rows",
                "duplicate_rows_after_first": n_dup,
                "duplicate_ratio": round(float(n_dup / len(df)), 6),
                "example_rows": examples,
            }
        ]
    )


# ---------------------------------------------------------
# E2) Relational duplicates: duplicate rows on subset keys (business key duplicates)
# ---------------------------------------------------------
def find_duplicate_rows_on_subsets(
    df: pd.DataFrame,
    table_name: str,
    cfg: Optional[DuplicateChecksConfig] = None,
    *,
    subset_keys: Optional[list[list[str]]] = None,
) -> pd.DataFrame:
    """
    Finds duplicates based on specified subsets of columns.
    Useful for business key duplicates like:
      payments: (order_id, amount, status)
    """
    cfg = cfg or DuplicateChecksConfig()
    if df is None or df.empty:
        return pd.DataFrame()

    if subset_keys is None:
        subset_keys = cfg.relational_duplicate_subset_keys.get(table_name, [])

    rows: list[dict[str, Any]] = []

    for subset in subset_keys or []:
        subset = _safe_cols(df, subset)
        if len(subset) == 0:
            continue

        dup_mask = df.duplicated(subset=subset, keep="first")
        n_dup = int(dup_mask.sum())
        if n_dup == 0:
            continue

        examples = df.loc[dup_mask, subset].head(cfg.max_examples).to_dict(orient="records")

        rows.append(
            {
                "table": table_name,
                "issue": "duplicate_rows_on_subset",
                "subset_cols": ", ".join(subset),
                "duplicate_rows_after_first": n_dup,
                "duplicate_ratio": round(float(n_dup / len(df)), 6),
                "example_subset_rows": examples,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# F) Relationship table checks: duplicate edges + cardinality violations
# ---------------------------------------------------------
def find_relationship_table_issues(
    df: pd.DataFrame,
    table_name: str,
    cfg: Optional[DuplicateChecksConfig] = None,
    *,
    left_col: Optional[str] = None,
    right_col: Optional[str] = None,
    expected: str = "unknown",
) -> pd.DataFrame:
    """
    For mapping or link tables, detect:
    - duplicate edges: same (left, right) pair repeated
    - cardinality violations:
        one_to_one: left maps to >1 right OR right maps to >1 left
        one_to_many: right maps to >1 left (unexpected)
        many_to_one: left maps to >1 right (unexpected)

    expected can be:
      "one_to_one", "one_to_many", "many_to_one", "many_to_many", "unknown"
    """
    cfg = cfg or DuplicateChecksConfig()
    if df is None or df.empty:
        return pd.DataFrame()

    if left_col is None or right_col is None:
        return pd.DataFrame()

    if left_col not in df.columns or right_col not in df.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    # Duplicate edges
    edge_dup_mask = df.duplicated(subset=[left_col, right_col], keep="first")
    n_edge_dup = int(edge_dup_mask.sum())
    if n_edge_dup > 0:
        examples = df.loc[edge_dup_mask, [left_col, right_col]].head(cfg.max_examples).to_dict(orient="records")
        rows.append(
            {
                "table": table_name,
                "issue": "duplicate_relationship_edges",
                "left_col": left_col,
                "right_col": right_col,
                "duplicate_edges_after_first": n_edge_dup,
                "duplicate_ratio": round(float(n_edge_dup / len(df)), 6),
                "example_edges": examples,
            }
        )

    # Cardinality checks
    x = df[[left_col, right_col]].dropna().copy()
    if x.empty:
        return pd.DataFrame(rows)

    left_to_right = x.groupby(left_col)[right_col].nunique()
    right_to_left = x.groupby(right_col)[left_col].nunique()

    left_multi = left_to_right[left_to_right > 1]
    right_multi = right_to_left[right_to_left > 1]

    def _add_cardinality_row(side: str, series: pd.Series, rule: str) -> None:
        examples = series.head(cfg.max_examples).to_dict()
        rows.append(
            {
                "table": table_name,
                "issue": "relationship_cardinality_violation",
                "expected": expected,
                "rule": rule,
                "side": side,
                "violating_keys_count": int(series.shape[0]),
                "example_key_to_distinct_count": examples,
            }
        )

    if expected == "one_to_one":
        if not left_multi.empty:
            _add_cardinality_row("left", left_multi, "left_maps_to_multiple_right")
        if not right_multi.empty:
            _add_cardinality_row("right", right_multi, "right_maps_to_multiple_left")

    elif expected == "one_to_many":
        # left is the one, right is the many
        if not right_multi.empty:
            _add_cardinality_row("right", right_multi, "right_maps_to_multiple_left")

    elif expected == "many_to_one":
        # left is the many, right is the one
        if not left_multi.empty:
            _add_cardinality_row("left", left_multi, "left_maps_to_multiple_right")

    elif expected in ("many_to_many", "unknown"):
        # still useful to report if extreme, but we keep it quiet by default
        pass

    return pd.DataFrame(rows)


def collapse_same_signature_multiple_ids(
    df: pd.DataFrame,
    cfg: DuplicateChecksConfig,
) -> pd.DataFrame:
    """
    Collapse redundant rows from find_same_natural_key_multiple_ids.

    Original output is one row per (table, id_column, signature_example).
    For reporting, we want one row per (table, signature_example), with:
      - which id columns were affected
      - the max distinct ids
      - a small sample of example rows
    """
    if df is None or df.empty:
        return pd.DataFrame()

    def _merge_examples(series: pd.Series) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in series.dropna().tolist():
            if not isinstance(item, list):
                continue
            for row in item:
                key = str(row)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(row)
                if len(merged) >= int(cfg.max_examples):
                    return merged
        return merged

    collapsed = (
        df.groupby(["table", "signature_example"], dropna=False)
        .agg(
            issue=("issue", "first"),
            signature_cols=("signature_cols", lambda s: ", ".join(sorted(set(s.dropna().astype(str))))),
            id_columns=("id_column", lambda s: ", ".join(sorted(set(s.dropna().astype(str))))),
            distinct_ids_for_signature=("distinct_ids_for_signature", "max"),
            example_rows=("example_rows", _merge_examples),
        )
        .reset_index()
    )

    # Optional: stable sort
    collapsed = collapsed.sort_values(
        ["table", "distinct_ids_for_signature"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return collapsed

def summarise_same_signature_multiple_ids(
    collapsed_df: pd.DataFrame,
    cfg: DuplicateChecksConfig,
) -> pd.DataFrame:
    """
    Produces a compact per-table summary:
    - how many signatures are problematic
    - max distinct ids seen for any signature
    - a few example signatures
    """
    if collapsed_df is None or collapsed_df.empty:
        return pd.DataFrame()

    def _example_sigs(s: pd.Series) -> str:
        vals = s.dropna().astype(str).unique().tolist()
        return ", ".join(vals[: int(cfg.max_examples)])

    out = (
        collapsed_df.groupby("table", dropna=False)
        .agg(
            issue=("issue", "first"),
            problem_signatures=("signature_example", "nunique"),
            max_distinct_ids=("distinct_ids_for_signature", "max"),
            affected_id_columns=("id_columns", lambda x: ", ".join(sorted(set(", ".join(x).split(", "))))),
            example_signatures=("signature_example", _example_sigs),
        )
        .reset_index()
        .sort_values(["problem_signatures", "max_distinct_ids"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return out


# ---------------------------------------------------------
# Master runner
# ---------------------------------------------------------
def run_duplicate_checks(
    dfs: dict[str, pd.DataFrame],
    cfg: Optional[DuplicateChecksConfig] = None,
) -> dict[str, pd.DataFrame]:
    cfg = cfg or DuplicateChecksConfig()

    dup_keys_all = []
    dup_natural_all = []
    id_reuse_all = []
    multi_ids_all = []

    exact_dups_all = []
    subset_dups_all = []
    relationship_issues_all = []

    for t, df in dfs.items():
        dup_keys_all.append(find_duplicate_values_in_key_columns(df, t, cfg))
        dup_natural_all.append(find_duplicate_values_in_natural_key_columns(df, t, cfg))
        id_reuse_all.append(find_within_table_id_value_reuse(df, t, cfg))
        multi_ids_all.append(find_same_natural_key_multiple_ids(df, t, cfg))

        exact_dups_all.append(find_exact_duplicate_rows(df, t, cfg))
        subset_dups_all.append(find_duplicate_rows_on_subsets(df, t, cfg))

        # relationship table rules if configured
        if t in cfg.relationship_tables:
            spec = cfg.relationship_tables[t]
            relationship_issues_all.append(
                find_relationship_table_issues(
                    df,
                    t,
                    cfg,
                    left_col=spec.get("left"),
                    right_col=spec.get("right"),
                    expected=spec.get("expected", "unknown"),
                )
            )

    # Cross-table overlaps
    overlap_same_name = find_cross_table_value_overlap_same_name(dfs, cfg)
    overlap_any_name = find_cross_table_value_overlap_id_and_natural(dfs, cfg)

    _raw_same_sig = pd.concat(multi_ids_all, ignore_index=True) if multi_ids_all else pd.DataFrame()
    _collapsed_same_sig = collapse_same_signature_multiple_ids(_raw_same_sig, cfg)


    return {
        "duplicate_keys": pd.concat(dup_keys_all, ignore_index=True) if dup_keys_all else pd.DataFrame(),
        "duplicate_natural_keys": pd.concat(dup_natural_all, ignore_index=True) if dup_natural_all else pd.DataFrame(),
        "cross_table_overlap_same_name": overlap_same_name,
        "cross_table_overlap_id_or_natural_any_name": overlap_any_name,
        "within_table_id_reuse": pd.concat(id_reuse_all, ignore_index=True) if id_reuse_all else pd.DataFrame(),
        "same_signature_multiple_ids_raw": _raw_same_sig,
        "same_signature_multiple_ids": _collapsed_same_sig,  # detailed per email
        "same_signature_multiple_ids_summary": summarise_same_signature_multiple_ids(_collapsed_same_sig, cfg),
        "exact_duplicate_rows": pd.concat(exact_dups_all, ignore_index=True) if exact_dups_all else pd.DataFrame(),
        "duplicate_rows_on_subsets": pd.concat(subset_dups_all, ignore_index=True) if subset_dups_all else pd.DataFrame(),
        "relationship_table_issues": pd.concat(relationship_issues_all, ignore_index=True) if relationship_issues_all else pd.DataFrame(),
    }

'''
Duplicate keys
Columns that look like IDs but contain repeated values that should be unique.

Duplicate natural keys
Repeated real-world identifiers such as emails, phone numbers, or usernames.

Cross-table overlap (same column name)
Columns with the same name in different tables that share many of the same values.

Cross-table overlap (different column names)
Identifier or natural-key columns in different tables that overlap even though their names differ.

ID value reuse within a table
The same identifier values appearing in multiple ID columns inside one table.

Same signature mapped to multiple IDs
The same real-world entity (for example same email or natural key) linked to more than one ID.

Summary of ID conflicts
A compact per-table summary showing how many identity conflicts exist and how severe they are.

Exact duplicate rows
Rows that are completely identical across all columns.

Duplicate rows on business keys
Rows that are duplicated when considering specific business key columns only.

Relationship table issues
Duplicate edges or unexpected cardinality violations in link or mapping tables.
'''