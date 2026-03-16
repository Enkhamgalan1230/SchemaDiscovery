"""
FK candidate scoring.

Takes unary IND candidates and assigns a confidence score using evidence:
- distinct coverage and row coverage (strongest signals)
- name similarity as a weak bonus (helps break ties)
- range penalty for int-like IDs to reduce accidental inclusions
  (e.g., small-domain values being included in a broad sequential parent key)

This module does not select final edges. It only scores candidates.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

# Tokens that appear in many "key" columns but carry almost no meaning.
_STOP_TOKENS = frozenset({
    "id", "no", "num", "number",
    "code", "key", "ref", "seq",
    "uid", "uuid", "guid",
})

@lru_cache(maxsize=2048)
def _tokens_cached(raw: str) -> frozenset[str]:
    raw = str(raw).strip()
    parts = re.split(r"[^A-Za-z0-9]+", raw)
    toks = []
    for p in parts:
        if not p:
            continue
        camel = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", p).split()
        toks.extend(camel)

    toks = [_norm_name(t) for t in toks if _norm_name(t)]
    # NEW: drop generic tokens
    toks = [t for t in toks if t not in _STOP_TOKENS]

    return frozenset(toks)

def name_similarity(a: str, b: str) -> float:
    na, nb = _norm_name(a), _norm_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0

    ta, tb = _tokens_cached(a), _tokens_cached(b)

    # if both became empty after stopwords, treat as no signal
    if not ta or not tb:
        return 0.0

    j = len(ta & tb) / len(ta | tb)

    sub = 0.65 if (na in nb or nb in na) else 0.0
    return float(max(j, sub))


def _range_penalty_from_minmax(
    df: pd.DataFrame,
    fk_distinct_small_max: int = 2000,
) -> pd.Series:
    """
    Compute a penalty in [0, 0.95] based on how much wider the parent range is
    compared to the child range.

    To reduce false positives, this penalty matters most when the FK domain is small.
    If fk_distinct is large, accidental inclusion is less likely, so we scale the
    penalty down.

    Only applies when fk_min/fk_max/pk_min/pk_max exist (int-like IDs).
    For string IDs those fields are null and the penalty is 0.
    """
    needed = ["fk_min", "fk_max", "pk_min", "pk_max"]
    mask = df[needed].notna().all(axis=1)

    pen = pd.Series(0.0, index=df.index, dtype="float64")
    if not mask.any():
        return pen

    sub = df.loc[mask]

    fk_range = (sub["fk_max"] - sub["fk_min"] + 1).astype(float).clip(lower=1.0)
    pk_range = (sub["pk_max"] - sub["pk_min"] + 1).astype(float).clip(lower=1.0)

    ratio = (pk_range / fk_range).clip(lower=1.0)
    logr = np.log10(ratio.to_numpy())

    # base penalty 0..1 with smooth saturation
    p = logr / (1.0 + logr)
    p = np.nan_to_num(p, nan=0.0, posinf=0.95, neginf=0.0)
    p = np.clip(p, 0.0, 0.95)

    # scale down penalty as fk_distinct grows
    if "fk_distinct" in sub.columns:
        fk_dist = sub["fk_distinct"].astype(float).clip(lower=1.0)
        # 1.0 when small, decays toward 0 as fk_distinct grows
        scale = np.minimum(1.0, float(fk_distinct_small_max) / fk_dist.to_numpy())
        p = p * scale

    pen.loc[mask] = p
    return pen.clip(0.0, 0.95)

def _small_domain_penalty(
    df: pd.DataFrame,
    *,
    fk_distinct_max: int = 30,
    distinct_cov_min: float = 0.95,
    min_name_sim_to_exempt: float = 0.30,
) -> pd.Series:
    """
    Penalise accidental inclusions from very small FK domains ONLY when the
    column name does not support it being an FK.

    This keeps true small FKs like SupplierID -> SupplierID.
    """
    fk_dist = df["fk_distinct"].astype(float)
    dc = df["distinct_coverage"].astype(float)

    # if name_sim exists, use it. If not, assume 0.
    ns = df["name_sim"].astype(float) if "name_sim" in df.columns else pd.Series(0.0, index=df.index)

    # penalise small domain only when:
    # -> small distinct
    # -> near perfect inclusion
    # -> and name similarity is weak
    bad = (fk_dist <= float(fk_distinct_max)) & (dc >= float(distinct_cov_min)) & (ns < float(min_name_sim_to_exempt))

    return bad.astype(float)

def _has_measure_tokens(col: str) -> bool:
    s = str(col).strip().lower()
    tokens = [
        "minute", "min", "time", "second",
        "day", "month", "year",
        "age", "score", "rating",
        "stage", "round", "week",
        "status", "flag", "type",
        "count", "qty", "quantity",
        "amount", "price", "total",
    ]
    return any(t in s for t in tokens)

def _looks_like_id(col: str) -> bool:
    s = str(col).strip().lower()
    return (s == "id") or s.endswith("id") or s.endswith("_id") or s.endswith("key_id")

def _measure_to_surrogate_penalty(
    df: pd.DataFrame,
    *,
    fk_max_small_cap: int = 500,
    fk_distinct_small_cap: int = 500,
    strong_cov: float = 0.99,
) -> pd.Series:
    """
    Penalise cases where a measure-like FK column (minute, status, month, etc.)
    is perfectly included in a surrogate-like parent key domain.
    """
    fk_name = df["fk_column"].astype("string")
    pk_name = df["pk_column"].astype("string")

    fk_measure = fk_name.map(_has_measure_tokens).fillna(False).astype("bool")
    fk_is_id   = fk_name.map(_looks_like_id).fillna(False).astype("bool")
    pk_is_id   = pk_name.map(_looks_like_id).fillna(False).astype("bool")

    d = df["distinct_coverage"].astype(float)
    r = df["row_coverage"].astype(float)

    # Use min/max when available, otherwise treat as not small bounded.
    fk_max = pd.to_numeric(df.get("fk_max"), errors="coerce")
    fk_dist = pd.to_numeric(df.get("fk_distinct"), errors="coerce")

    small_bounded = (fk_max.notna()) & (fk_max <= float(fk_max_small_cap)) & (fk_dist.notna()) & (fk_dist <= float(fk_distinct_small_cap))
    perfectish = (d >= float(strong_cov)) & (r >= float(strong_cov))

    bad = fk_measure & (~fk_is_id) & pk_is_id & small_bounded & perfectish
    return bad.astype(float)


def score_ind_candidates(
    ind_df: pd.DataFrame,
    w_distinct: float = 0.75,
    w_row: float = 0.20,
    w_name: float = 0.05,
    w_range_penalty: float = 0.10,
    fk_distinct_small_max: int = 2000,
    w_small_domain_penalty: float = 0.20,
    small_fk_distinct_max: int = 30,
    small_distinct_cov_min: float = 0.95,
    w_measure_penalty: float = 0.60,
) -> pd.DataFrame:
    required = {
        "fk_table", "fk_column", "pk_table", "pk_column",
        "distinct_coverage", "row_coverage",
        "fk_distinct", "pk_distinct", "intersection_distinct",
        "fk_min", "fk_max", "pk_min", "pk_max",
    }
    missing = required - set(ind_df.columns)
    if missing:
        raise ValueError(f"ind_df missing required columns: {sorted(missing)}")

    # basic weight sanity checks (no auto-normalisation to avoid changing behaviour)
    for name, w in [
        ("w_distinct", w_distinct),
        ("w_row", w_row),
        ("w_name", w_name),
        ("w_range_penalty", w_range_penalty),
    ]:
        if float(w) < 0.0:
            raise ValueError(f"{name} must be >= 0")

    df = ind_df.copy()

    # weak semantic signal (bonus, not a gate)
    df["name_sim"] = [
        name_similarity(fk, pk) for fk, pk in zip(df["fk_column"], df["pk_column"])
    ]

    # penalty against accidental inclusions (only for int-like min/max rows)
    df["range_penalty"] = _range_penalty_from_minmax(df, fk_distinct_small_max=fk_distinct_small_max)

    df["measure_penalty"] = _measure_to_surrogate_penalty(df)

    # ensure numeric
    d = df["distinct_coverage"].astype(float)
    r = df["row_coverage"].astype(float)
    ns = df["name_sim"].astype(float)
    rp = df["range_penalty"].astype(float)

    df["small_domain_penalty"] = _small_domain_penalty(
        df,
        fk_distinct_max=small_fk_distinct_max,
        distinct_cov_min=small_distinct_cov_min,
    )

    sdp = df["small_domain_penalty"].astype(float)

    mp = df["measure_penalty"].astype(float)

    df["score"] = (
        float(w_distinct) * d
        + float(w_row) * r
        + float(w_name) * ns
        - float(w_range_penalty) * rp
        - float(w_small_domain_penalty) * sdp
        - float(w_measure_penalty) * mp
    ).clip(0.0, 1.0)

    return (
        df.sort_values(
            by=["fk_table", "fk_column", "score", "distinct_coverage", "row_coverage", "name_sim"],
            ascending=[True, True, False, False, False, False],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
