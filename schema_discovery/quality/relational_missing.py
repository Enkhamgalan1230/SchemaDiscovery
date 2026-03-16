"""
Relational missingness checks + classification.

Two core checks per discovered relationship edge (FK -> PK):

1) Orphan FK
   Child has FK values not present in parent PK domain.
   Condition -> FK_value ∉ PK_set (for non-null FK rows)

2) Missing children
   Parent has PK values that are not present in the child FK domain.
   Condition -> PK_value ∉ FK_set (for parent keys)

We also provide a lightweight, explainable classification:
- status: ok | warning | fail
- severity: low | medium | high
- likely_causes: list[str]
- recommended_actions: list[str]

This module is intentionally rule-based (not ML) so its outputs are debuggable
and safe to expose via API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from schema_discovery.quality.key_normalisation import norm_key_series


# -----------------------------
# Configuration
# -----------------------------
@dataclass(frozen=True)
class RelMissingConfig:
    orphan_warn_ratio: float = 0.01
    orphan_fail_ratio: float = 0.05

    missing_warn_ratio: float = 0.05
    missing_fail_ratio: float = 0.20

    required_multiplier: float = 1.0

    sentinel_max_examples_scan: int = 50
    sentinel_strings: tuple[str, ...] = ("unknown", "unk", "na", "n/a", "null", "none", "missing")
    sentinel_ints: tuple[int, ...] = (-1, 0, 999, 9999, 99999, 999999, 888888)


# -----------------------------
# Helpers
# -----------------------------
def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame")
    return df[col]


def _to_set(s: pd.Series) -> set:
    if s is None or len(s) == 0:
        return set()
    return set(s.tolist())


def _sample_unique(values: list[Any], n: int) -> list[Any]:
    if not values:
        return []
    return list(pd.Series(values).drop_duplicates().head(int(n)).tolist())


def _detect_sentinel(values: list[Any], cfg: RelMissingConfig) -> tuple[bool, list[str]]:
    """
    Return (is_sentinel_like, reasons)
    Based on examples of orphan FK values or suspicious FK values.
    """
    reasons: list[str] = []
    if not values:
        return False, reasons

    scan = values[: int(cfg.sentinel_max_examples_scan)]

    # normalize to strings for string sentinel check
    lowered = []
    for v in scan:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        lowered.append(str(v).strip().lower())

    # string sentinels
    for tok in cfg.sentinel_strings:
        if any(tok == s for s in lowered):
            reasons.append(f"Contains sentinel string '{tok}'")
            break

    # int sentinels (handle "999999.0" etc)
    ints_found = set()
    for v in scan:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        try:
            iv = int(float(v))
            ints_found.add(iv)
        except Exception:
            continue

    hit_ints = [x for x in cfg.sentinel_ints if x in ints_found]
    if hit_ints:
        reasons.append(f"Contains sentinel numeric(s) {hit_ints[:5]}")

    return (len(reasons) > 0), reasons


def _profile_lookup(profiles_df: pd.DataFrame, table: str, col: str) -> dict[str, Any]:
    """
    Lightweight profile row lookup. Returns empty dict if not found.
    """
    if profiles_df is None or profiles_df.empty:
        return {}
    try:
        sub = profiles_df[(profiles_df["table_name"] == table) & (profiles_df["column_name"] == col)]
        if sub.empty:
            return {}
        row = sub.iloc[0].to_dict()
        return row
    except Exception:
        return {}


def _status_from_ratios(
    *,
    orphan_ratio: float,
    missing_ratio: float,
    optional: Optional[bool],
    cfg: RelMissingConfig,
) -> tuple[str, str]:
    """
    Return (status, severity).
    Uses stricter thresholds when optional is False (required relationship).
    """
    mult = 1.0
    if optional is False:
        mult = float(cfg.required_multiplier)

    orphan_warn = cfg.orphan_warn_ratio * mult
    orphan_fail = cfg.orphan_fail_ratio * mult
    missing_warn = cfg.missing_warn_ratio * mult
    missing_fail = cfg.missing_fail_ratio * mult

    # compute worst-case severity
    # fail if either ratio crosses fail threshold
    if (orphan_ratio >= orphan_fail) or (missing_ratio >= missing_fail):
        return "fail", "high"

    # warning if either crosses warning threshold
    if (orphan_ratio >= orphan_warn) or (missing_ratio >= missing_warn):
        return "warning", "medium"

    return "ok", "low"


# -----------------------------
# Core checks
# -----------------------------
def check_orphan_fk(
    *,
    child_df: pd.DataFrame,
    child_table: str,
    child_fk_col: str,
    parent_df: pd.DataFrame,
    parent_table: str,
    parent_pk_col: str,
    sample_n: int = 10,
    include_row_indices: bool = True,
) -> dict[str, Any]:
    """
    Orphan FK -> child has FK values not in parent PK set.

    Returns counts, ratio, example values, and optionally child row indices where orphan occurs.
    """
    fk_raw = _safe_series(child_df, child_fk_col)
    fk_norm = norm_key_series(fk_raw)
    fk_non_null_rows = int(len(fk_norm))

    child_total_rows = int(len(child_df))
    fk_null_rows = int(child_total_rows - fk_non_null_rows)
    fk_null_ratio = (fk_null_rows / child_total_rows) if child_total_rows else 0.0

    pk_norm = norm_key_series(_safe_series(parent_df, parent_pk_col))
    pk_set = _to_set(pk_norm)

    fk_vals = fk_norm.tolist()
    orphan_mask = [v not in pk_set for v in fk_vals]
    orphan_vals = [v for v, is_orphan in zip(fk_vals, orphan_mask) if is_orphan]

    orphan_rows = int(len(orphan_vals))
    orphan_ratio = (orphan_rows / fk_non_null_rows) if fk_non_null_rows else 0.0

    examples = _sample_unique(orphan_vals, n=sample_n)

    out: dict[str, Any] = {
        "check": "orphan_fk",
        "child": {"table": child_table, "fk_column": child_fk_col},
        "parent": {"table": parent_table, "pk_column": parent_pk_col},

        "child_total_rows": child_total_rows,
        "fk_non_null_rows": fk_non_null_rows,
        "fk_null_rows": fk_null_rows,
        "fk_null_ratio": round(float(fk_null_ratio), 6),

        "orphan_fk_rows": orphan_rows,
        "orphan_fk_ratio": round(float(orphan_ratio), 6),
        "orphan_fk_examples": examples,
    }

    if include_row_indices and orphan_rows > 0:
        # Row indices in child_df corresponding to orphan fk values
        # We need indices aligned to fk_norm, so use mask on normalized series index
        # norm_key_series typically drops nulls and preserves original indices -> ideal
        orphan_idx = fk_norm.index[np.array(orphan_mask, dtype=bool)]
        out["orphan_child_row_indices"] = orphan_idx[:sample_n].tolist()

    return out


def check_missing_children(
    *,
    parent_df: pd.DataFrame,
    parent_table: str,
    parent_pk_col: str,
    child_df: pd.DataFrame,
    child_table: str,
    child_fk_col: str,
    sample_n: int = 10,
) -> dict[str, Any]:
    """
    Missing children -> parent PK values not present in child FK set.

    Note:
    This does NOT mean "other cells in that child row are missing".
    It means "there exists no child row referencing that parent PK".
    """
    parent_norm = norm_key_series(_safe_series(parent_df, parent_pk_col))
    parent_set = _to_set(parent_norm)

    child_norm = norm_key_series(_safe_series(child_df, child_fk_col))
    child_set = _to_set(child_norm)

    missing_parents = [pk for pk in parent_set if pk not in child_set]

    missing_count = int(len(missing_parents))
    parent_count = int(len(parent_set))
    missing_ratio = (missing_count / parent_count) if parent_count else 0.0

    examples = _sample_unique(missing_parents, n=sample_n)

    return {
        "check": "missing_children",
        "parent": {"table": parent_table, "pk_column": parent_pk_col},
        "child": {"table": child_table, "fk_column": child_fk_col},
        "parent_rows": parent_count,
        "parents_missing_children": missing_count,
        "parents_missing_children_ratio": round(float(missing_ratio), 6),
        "parent_pk_examples": examples,
    }


# -----------------------------
# Classification
# -----------------------------
def classify_rel_missing(
    *,
    edge: dict[str, Any],
    orphan_result: dict[str, Any],
    missing_result: dict[str, Any],
    profiles_df: Optional[pd.DataFrame] = None,
    cfg: Optional[RelMissingConfig] = None,
) -> dict[str, Any]:
    """
    Combine check outputs + profiles evidence into an explainable classification.

    `edge` should minimally include:
      fk_table, fk_column, pk_table, pk_column
    Optional:
      optional (bool), cardinality, score, name_sim, range_penalty
    """
    cfg = cfg or RelMissingConfig()

    fk_table = str(edge["fk_table"])
    fk_col = str(edge["fk_column"])
    pk_table = str(edge["pk_table"])
    pk_col = str(edge["pk_column"])

    optional = edge.get("optional", None)

    orphan_ratio = float(orphan_result.get("orphan_fk_ratio", 0.0) or 0.0)
    missing_ratio = float(missing_result.get("parents_missing_children_ratio", 0.0) or 0.0)

    status, severity = _status_from_ratios(
        orphan_ratio=orphan_ratio,
        missing_ratio=missing_ratio,
        optional=optional if isinstance(optional, bool) else None,
        cfg=cfg,
    )

    likely_causes: list[str] = []
    notes: list[str] = []
    actions: list[str] = []

    # Sentinel / default detection for orphans
    orphan_examples = orphan_result.get("orphan_fk_examples", []) or []
    is_sentinel, sentinel_reasons = _detect_sentinel(orphan_examples, cfg)
    if orphan_ratio > 0 and is_sentinel:
        likely_causes.append("sentinel_or_default_fk_values")
        notes.extend(sentinel_reasons)
        actions.append("Treat sentinel FK values as 'unknown' and exclude them from orphan metrics or report separately.")

    # Optional relationship evidence (from profiles_df null_ratio)
    prof = _profile_lookup(profiles_df, fk_table, fk_col) if profiles_df is not None else {}
    fk_null_ratio = prof.get("null_ratio", None)
    if fk_null_ratio is not None:
        try:
            fk_null_ratio = float(fk_null_ratio)
        except Exception:
            fk_null_ratio = None

    if missing_ratio > 0:
        if optional is True or (fk_null_ratio is not None and fk_null_ratio >= 0.01):
            likely_causes.append("relationship_optional_or_sparse_fk")
            actions.append("If business rules allow, downgrade severity for missing children or mark edge as optional.")
        else:
            likely_causes.append("child_table_incomplete_or_filtered")
            actions.append("Check if child table extract is filtered (date window, sampling) or missing batches.")

    if orphan_ratio > 0:
        # possible type mismatch normalisation issues
        likely_causes.append("type_or_format_mismatch_or_missing_parent_rows")
        actions.append("Confirm FK and PK are normalised to comparable types and parent table includes all referenced keys.")

    # Weak edge evidence: name similarity and range penalty (if present)
    name_sim = edge.get("name_sim", None)
    range_penalty = edge.get("range_penalty", None)
    score = edge.get("score", None)

    try:
        if score is not None and float(score) < 0.95 and (orphan_ratio > 0 or missing_ratio > 0):
            likely_causes.append("edge_may_be_wrong_or_accidental_match")
            actions.append("Consider raising min_edge_score or requiring higher name similarity for accepting edges.")
    except Exception:
        pass

    try:
        if name_sim is not None and float(name_sim) < 0.2 and (orphan_ratio > 0):
            likely_causes.append("low_name_similarity_possible_false_positive")
            actions.append("Review column naming or require name_sim threshold for FK selection.")
    except Exception:
        pass

    try:
        if range_penalty is not None and float(range_penalty) > 0.5 and (orphan_ratio > 0):
            likely_causes.append("range_mismatch_possible_accidental_inclusion")
            actions.append("Review range penalty settings or increase distinct coverage threshold to reduce false positives.")
    except Exception:
        pass

    # de-duplicate while preserving order
    likely_causes = list(dict.fromkeys(likely_causes))
    notes = list(dict.fromkeys(notes))
    actions = list(dict.fromkeys(actions))

    return {
        "status": status,
        "severity": severity,
        "likely_causes": likely_causes,
        "notes": notes,
        "recommended_actions": actions,
    }


def run_rel_missing_for_edge(
    *,
    dfs: dict[str, pd.DataFrame],
    edge_row: pd.Series | dict[str, Any],
    profiles_df: Optional[pd.DataFrame] = None,
    cfg: Optional[RelMissingConfig] = None,
    sample_n: int = 10,
) -> dict[str, Any]:
    """
    Convenience wrapper: runs both checks + classification for one edge.
    Returns a single JSON-safe dict.
    """
    cfg = cfg or RelMissingConfig()
    edge = dict(edge_row)

    fk_table = str(edge["fk_table"])
    fk_col = str(edge["fk_column"])
    pk_table = str(edge["pk_table"])
    pk_col = str(edge["pk_column"])

    child_df = dfs[fk_table]
    parent_df = dfs[pk_table]

    orphan = check_orphan_fk(
        child_df=child_df,
        child_table=fk_table,
        child_fk_col=fk_col,
        parent_df=parent_df,
        parent_table=pk_table,
        parent_pk_col=pk_col,
        sample_n=sample_n,
        include_row_indices=True,
    )

    missing = check_missing_children(
        parent_df=parent_df,
        parent_table=pk_table,
        parent_pk_col=pk_col,
        child_df=child_df,
        child_table=fk_table,
        child_fk_col=fk_col,
        sample_n=sample_n,
    )

    classification = classify_rel_missing(
        edge=edge,
        orphan_result=orphan,
        missing_result=missing,
        profiles_df=profiles_df,
        cfg=cfg,
    )

    return {
        "edge": f"{fk_table}.{fk_col} -> {pk_table}.{pk_col}",
        "edge_ref": {"fk_table": fk_table, "fk_column": fk_col, "pk_table": pk_table, "pk_column": pk_col},
        "checks": {"orphan_fk": orphan, "missing_children": missing},
        "classification": classification,
    }


def rel_missing_summary_to_df(
    results: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Convert list of run_rel_missing_for_edge outputs into a flat DataFrame
    suitable for pipeline artifacts / API stage output.
    """
    rows: list[dict[str, Any]] = []
    for item in results:
        edge = item.get("edge")
        ref = item.get("edge_ref", {})
        orphan = (item.get("checks", {}) or {}).get("orphan_fk", {}) or {}
        missing = (item.get("checks", {}) or {}).get("missing_children", {}) or {}
        cls = item.get("classification", {}) or {}

        rows.append(
            {
                "relationship": edge,
                "fk_table": ref.get("fk_table"),
                "fk_column": ref.get("fk_column"),
                "pk_table": ref.get("pk_table"),
                "pk_column": ref.get("pk_column"),

                "child_total_rows": orphan.get("child_total_rows", 0),
                "fk_non_null_rows": orphan.get("fk_non_null_rows", 0),
                "fk_null_rows": orphan.get("fk_null_rows", 0),
                "fk_null_ratio": orphan.get("fk_null_ratio", 0.0),

                "orphan_fk_rows": orphan.get("orphan_fk_rows", 0),
                "orphan_fk_ratio": orphan.get("orphan_fk_ratio", 0.0),

                "unreferenced_parent_keys": missing.get("parents_missing_children", 0),
                "unreferenced_parent_ratio": missing.get("parents_missing_children_ratio", 0.0),

                "status": cls.get("status"),
                "severity": cls.get("severity"),
                "likely_causes": ", ".join(cls.get("likely_causes", []) or []),
                "recommended_actions": ", ".join(cls.get("recommended_actions", []) or []),
            }
        )


    return pd.DataFrame(rows)
