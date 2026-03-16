from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Iterable

import numpy as np
import pandas as pd

from schema_discovery.quality.key_normalisation import norm_key_series


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class DefaultFKConfig:
    # How many of the most frequent FK values to inspect
    top_k: int = 3

    # Minimum share of FK rows for a value to be suspicious
    min_ratio: float = 0.20

    # Very strong share threshold (even if value is not a known sentinel)
    strong_ratio: float = 0.50

    # If FK has very small distinct domain, it is likely a categorical code
    # We skip those unless sentinel evidence exists OR pairing variance is strong
    small_domain_max_distinct: int = 5

    # Minimum FK non null rows before we try to infer anything
    min_non_null_rows: int = 20

    # Sentinel lists
    sentinel_strings: tuple[str, ...] = (
        "unknown", "unk",
        "na", "n/a", "n.a.", "nan",
        "null", "none", "nil",
        "missing",
        "undefined", "unset", "not_set", "notset",
        "not applicable", "not_applicable",
        "not available", "not_available",
        "default", "other", "misc",
        "tbd", "todo",
        "?", "??",
    )

    sentinel_ints: tuple[int, ...] = (
        -1, 0,
        99, 999, 9999, 99999, 999999,
        88, 8888, 888888,
    )

    # Dummy parent row check (optional, off by default)
    enable_dummy_parent_check: bool = False
    parent_label_cols: Optional[tuple[str, ...]] = None
    dummy_parent_tokens: tuple[str, ...] = ("unknown", "default", "misc", "na", "n/a")

    # -----------------------------
    # Pairing variance check (NEW)
    # -----------------------------
    enable_pairing_variance_check: bool = True

    # Which columns in the child table count as "partner identifiers"
    # -> if True, only *_id / id / uuid-like columns are considered partners
    partner_id_like_only: bool = True

    # Natural key patterns also allowed as partners (if partner_id_like_only=False)
    partner_natural_key_substrings: tuple[str, ...] = ("email", "e_mail", "phone", "mobile", "msisdn", "username")

    # Gate: partner column needs enough overall distinct values to be meaningful
    min_partner_overall_distinct: int = 20

    # For a given FK value v, how many distinct partners must it connect to?
    min_partner_distinct_for_flag: int = 10

    # Ratio version: partner_distinct_under_v / overall_partner_distinct
    min_partner_distinct_ratio: float = 0.20

    # Max examples to include
    max_partner_examples: int = 10


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


def _safe_value_str(v: Any) -> str:
    try:
        return str(v)
    except Exception:
        return "<unprintable>"


def _as_int_if_possible(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return int(float(v))
    except Exception:
        return None


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


def _is_natural_key_like(col: str, cfg: DefaultFKConfig) -> bool:
    c = _norm_colname(col)
    return any(pat in c for pat in cfg.partner_natural_key_substrings)


def _is_sentinel_value(v: Any, cfg: DefaultFKConfig) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    sv = _safe_value_str(v).strip().lower()
    if sv in cfg.sentinel_strings:
        reasons.append(f"sentinel_string:{sv}")

    iv = _as_int_if_possible(v)
    if iv is not None and iv in cfg.sentinel_ints:
        reasons.append(f"sentinel_int:{iv}")

    return (len(reasons) > 0), reasons


def _dummy_parent_semantic_hit(
    *,
    parent_df: pd.DataFrame,
    parent_pk_col: str,
    pk_value: Any,
    cfg: DefaultFKConfig,
) -> tuple[bool, list[str]]:
    if not cfg.enable_dummy_parent_check:
        return False, []
    if not cfg.parent_label_cols:
        return False, []

    cols = [c for c in cfg.parent_label_cols if c in parent_df.columns]
    if not cols:
        return False, []

    try:
        mask = parent_df[parent_pk_col].astype(object) == pk_value
        sub = parent_df.loc[mask, cols]
        if sub.empty:
            return False, []
    except Exception:
        return False, []

    tokens = set(cfg.dummy_parent_tokens)
    reasons: list[str] = []

    try:
        sample = sub.head(5)
        for c in cols:
            for v in sample[c].tolist():
                txt = _safe_value_str(v).strip().lower()
                if not txt:
                    continue
                for tok in tokens:
                    if tok == txt or tok in txt:
                        reasons.append(f"dummy_parent_token:{tok} in {c}")
                        return True, reasons
    except Exception:
        return False, []

    return False, []


def _pick_partner_cols(
    *,
    child_df: pd.DataFrame,
    fk_col: str,
    cfg: DefaultFKConfig,
) -> list[str]:
    """
    Partner cols are other identifier-like columns in the same child table.
    These columns help detect "value v pairs with many different entities".
    """
    cols = [str(c) for c in child_df.columns if str(c) != str(fk_col)]

    if cfg.partner_id_like_only:
        return [c for c in cols if _is_id_like(c)]

    # otherwise allow both id-like and natural keys
    out: list[str] = []
    for c in cols:
        if _is_id_like(c) or _is_natural_key_like(c, cfg):
            out.append(c)
    return out


def _pairing_variance_evidence(
    *,
    child_df: pd.DataFrame,
    fk_col: str,
    fk_value: Any,
    partner_cols: list[str],
    cfg: DefaultFKConfig,
) -> tuple[bool, list[str], dict[str, Any]]:
    """
    For rows where FK==fk_value, see if it pairs with many distinct partner IDs.

    Returns:
    (flagged, reasons, evidence)
    """
    if not cfg.enable_pairing_variance_check:
        return False, [], {}

    if not partner_cols:
        return False, [], {}

    # Select rows where fk_col equals fk_value (use object compare for stability)
    try:
        fk_s = child_df[fk_col].astype(object)
        mask = fk_s == fk_value
        sub = child_df.loc[mask, partner_cols]
        if sub.empty:
            return False, [], {}
    except Exception:
        return False, [], {}

    best_partner = None
    best_distinct = 0
    best_ratio = 0.0
    best_examples: list[Any] = []

    # Evaluate each partner column
    for pcol in partner_cols:
        try:
            p_norm = norm_key_series(sub[pcol])
            distinct_under_v = int(p_norm.nunique(dropna=True))
            if distinct_under_v <= 0:
                continue

            overall_norm = norm_key_series(child_df[pcol])
            overall_distinct = int(overall_norm.nunique(dropna=True))

            if overall_distinct < int(cfg.min_partner_overall_distinct):
                continue

            ratio = float(distinct_under_v / overall_distinct) if overall_distinct else 0.0

            # track best
            if distinct_under_v > best_distinct or (distinct_under_v == best_distinct and ratio > best_ratio):
                best_partner = pcol
                best_distinct = distinct_under_v
                best_ratio = ratio
                best_examples = p_norm.drop_duplicates().head(int(cfg.max_partner_examples)).tolist()
        except Exception:
            continue

    if best_partner is None:
        return False, [], {}

    flagged = (
        best_distinct >= int(cfg.min_partner_distinct_for_flag)
        and best_ratio >= float(cfg.min_partner_distinct_ratio)
    )

    if not flagged:
        return False, [], {
            "best_partner_column": best_partner,
            "distinct_partners_under_value": best_distinct,
            "partner_distinct_ratio": round(float(best_ratio), 6),
        }

    reasons = [
        "pairs_with_many_distinct_partner_ids",
        f"partner_col:{best_partner}",
    ]

    evidence = {
        "best_partner_column": best_partner,
        "distinct_partners_under_value": int(best_distinct),
        "partner_distinct_ratio": round(float(best_ratio), 6),
        "partner_examples": best_examples,
    }

    return True, reasons, evidence


# -----------------------------
# Core check
# -----------------------------
def check_default_fk_values(
    *,
    child_df: pd.DataFrame,
    child_table: str,
    child_fk_col: str,
    parent_df: pd.DataFrame,
    parent_table: str,
    parent_pk_col: str,
    cfg: Optional[DefaultFKConfig] = None,
    sample_n: int = 10,
) -> dict[str, Any]:
    cfg = cfg or DefaultFKConfig()

    fk_raw = _safe_series(child_df, child_fk_col)
    pk_raw = _safe_series(parent_df, parent_pk_col)

    fk_norm = norm_key_series(fk_raw)
    pk_norm = norm_key_series(pk_raw)

    fk_non_null_rows = int(len(fk_norm))
    pk_set = _to_set(pk_norm)

    relationship = f"{child_table}.{child_fk_col} -> {parent_table}.{parent_pk_col}"

    out: dict[str, Any] = {
        "check": "default_fk_values",
        "relationship": relationship,
        "edge_ref": {
            "fk_table": str(child_table),
            "fk_column": str(child_fk_col),
            "pk_table": str(parent_table),
            "pk_column": str(parent_pk_col),
        },
        "fk_non_null_rows": fk_non_null_rows,
        "fk_distinct": int(fk_norm.nunique(dropna=True)) if fk_non_null_rows else 0,
        "default_detected": False,
        "candidates": [],
    }

    if fk_non_null_rows < int(cfg.min_non_null_rows):
        return out

    fk_distinct = int(out["fk_distinct"])

    vc = fk_norm.value_counts(dropna=True)
    top_k = int(max(1, cfg.top_k))
    items = list(vc.head(top_k).items())
    if not items:
        return out

    partner_cols = _pick_partner_cols(child_df=child_df, fk_col=child_fk_col, cfg=cfg)

    for v, cnt in items:
        cnt = int(cnt)
        ratio = float(cnt / fk_non_null_rows) if fk_non_null_rows else 0.0

        in_parent = bool(v in pk_set)
        is_sentinel, sentinel_reasons = _is_sentinel_value(v, cfg)

        reasons: list[str] = []
        reasons.extend(sentinel_reasons)

        flagged = False
        evidence: dict[str, Any] = {}

        # Rule 1: orphan placeholder with meaningful frequency
        if ratio >= float(cfg.min_ratio) and (not in_parent):
            flagged = True
            reasons.append("high_frequency_orphan_value")

        # Rule 2: sentinel with meaningful frequency
        if ratio >= float(cfg.min_ratio) and is_sentinel:
            flagged = True
            reasons.append("high_frequency_sentinel_value")

        # Rule 3: extreme bucket share (avoid small domains)
        if ratio >= float(cfg.strong_ratio) and fk_distinct > int(cfg.small_domain_max_distinct):
            flagged = True
            reasons.append("extreme_bucket_share")

        # Rule 4: pairing variance (NEW)
        # Even if FK is in parent domain, a default can still exist (dummy row).
        if ratio >= float(cfg.min_ratio):
            pv_flag, pv_reasons, pv_evidence = _pairing_variance_evidence(
                child_df=child_df,
                fk_col=child_fk_col,
                fk_value=v,
                partner_cols=partner_cols,
                cfg=cfg,
            )
            if pv_flag:
                flagged = True
                reasons.extend(pv_reasons)
                evidence.update(pv_evidence)

        # Small domain guard:
        # If FK domain is tiny, treat it as categorical unless sentinel or pairing variance is strong
        if fk_distinct <= int(cfg.small_domain_max_distinct) and (not is_sentinel) and ("pairs_with_many_distinct_partner_ids" not in reasons):
            flagged = False
            reasons = []
            evidence = {}

        # Optional dummy parent semantic check
        if flagged and in_parent:
            hit, dummy_reasons = _dummy_parent_semantic_hit(
                parent_df=parent_df,
                parent_pk_col=parent_pk_col,
                pk_value=v,
                cfg=cfg,
            )
            if hit:
                reasons.extend(dummy_reasons)
                reasons.append("dummy_parent_row_semantics")

        if not flagged:
            continue

        item: dict[str, Any] = {
            "value": v,
            "count": cnt,
            "ratio": round(ratio, 6),
            "in_parent_pk_domain": in_parent,
            "reasons": list(dict.fromkeys(reasons))[:10],
        }
        if evidence:
            item["evidence"] = evidence

        out["candidates"].append(item)

    out["default_detected"] = len(out["candidates"]) > 0

    if out["default_detected"] and sample_n > 0:
        out["examples"] = [c["value"] for c in out["candidates"][: int(min(sample_n, len(out["candidates"])))]]

    return out


# -----------------------------
# Batch runner for edges
# -----------------------------
def run_default_fk_for_edges(
    *,
    dfs: dict[str, pd.DataFrame],
    edges_df: pd.DataFrame,
    cfg: Optional[DefaultFKConfig] = None,
    sample_n: int = 10,
) -> list[dict[str, Any]]:
    cfg = cfg or DefaultFKConfig()

    required = {"fk_table", "fk_column", "pk_table", "pk_column"}
    missing = required - set(edges_df.columns)
    if missing:
        raise ValueError(f"edges_df missing required columns: {sorted(missing)}")

    results: list[dict[str, Any]] = []

    for r in edges_df.itertuples(index=False):
        fk_table = str(getattr(r, "fk_table"))
        fk_col = str(getattr(r, "fk_column"))
        pk_table = str(getattr(r, "pk_table"))
        pk_col = str(getattr(r, "pk_column"))

        if fk_table not in dfs or pk_table not in dfs:
            continue
        if fk_col not in dfs[fk_table].columns or pk_col not in dfs[pk_table].columns:
            continue

        results.append(
            check_default_fk_values(
                child_df=dfs[fk_table],
                child_table=fk_table,
                child_fk_col=fk_col,
                parent_df=dfs[pk_table],
                parent_table=pk_table,
                parent_pk_col=pk_col,
                cfg=cfg,
                sample_n=sample_n,
            )
        )

    return results


# -----------------------------
# Flatten to DataFrame (pipeline friendly)
# -----------------------------
def default_fk_summary_to_df(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for item in results or []:
        ref = item.get("edge_ref", {}) or {}
        rel = item.get("relationship")
        detected = bool(item.get("default_detected", False))
        fk_non_null = int(item.get("fk_non_null_rows", 0) or 0)
        fk_distinct = int(item.get("fk_distinct", 0) or 0)

        candidates = item.get("candidates", []) or []
        try:
            candidates = sorted(candidates, key=lambda x: float(x.get("ratio", 0.0) or 0.0), reverse=True)
        except Exception:
            pass

        top_value = None
        top_ratio = None
        top_count = None
        top_in_parent = None
        top_reasons = None
        top_partner_col = None
        top_partner_distinct = None
        top_partner_ratio = None

        if candidates:
            top = candidates[0]
            top_value = top.get("value")
            top_ratio = top.get("ratio")
            top_count = top.get("count")
            top_in_parent = top.get("in_parent_pk_domain")
            rs = top.get("reasons", []) or []
            top_reasons = ", ".join([str(x) for x in rs[:8]])

            ev = top.get("evidence", {}) or {}
            top_partner_col = ev.get("best_partner_column")
            top_partner_distinct = ev.get("distinct_partners_under_value")
            top_partner_ratio = ev.get("partner_distinct_ratio")

        rows.append(
            {
                "relationship": rel,
                "fk_table": ref.get("fk_table"),
                "fk_column": ref.get("fk_column"),
                "pk_table": ref.get("pk_table"),
                "pk_column": ref.get("pk_column"),
                "default_detected": detected,
                "fk_non_null_rows": fk_non_null,
                "fk_distinct": fk_distinct,
                "top_default_value": top_value,
                "top_default_ratio": top_ratio,
                "top_default_count": top_count,
                "top_default_in_parent_domain": top_in_parent,
                "top_default_reasons": top_reasons,
                "top_partner_column": top_partner_col,
                "top_partner_distinct_under_value": top_partner_distinct,
                "top_partner_distinct_ratio": top_partner_ratio,
                "num_candidates": int(len(candidates)),
            }
        )

    return pd.DataFrame(rows)