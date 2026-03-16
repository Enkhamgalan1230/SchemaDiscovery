from __future__ import annotations

from typing import Any

import pandas as pd

from ..disclosure import disclose_key_normalisation
from ..schema import AltOption, Blocker, RecommendationRecord
from ..utils import clamp01, safe_float, top_n
from .. import reasons as R


def _profile_row(profiles_df: pd.DataFrame, table: str, col: str) -> dict[str, Any]:
    if profiles_df is None or profiles_df.empty:
        return {}
    sub = profiles_df[(profiles_df["table_name"] == table) & (profiles_df["column_name"] == col)]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


def recommend_primary_key(
    *,
    table_name: str,
    profiles_df: pd.DataFrame,
    ucc_df: pd.DataFrame,
    composite_ucc_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    surrogate_keys_out: dict[str, Any],
    dup_out: dict[str, pd.DataFrame],
    top_n_alternatives: int,
) -> dict[str, Any]:
    # Candidates from unary UCC
    candidates: list[dict[str, Any]] = []
    if ucc_df is not None and not ucc_df.empty:
        sub = ucc_df[ucc_df["table_name"] == table_name] if "table_name" in ucc_df.columns else pd.DataFrame()
        for r in sub.to_dict(orient="records"):
            col = str(r.get("column_name"))
            prof = _profile_row(profiles_df, table_name, col)
            nr = safe_float(prof.get("null_ratio"), default=None)
            null_ratio = 1.0 if nr is None else nr
            unique_ratio = safe_float(prof.get("unique_ratio_non_null", prof.get("unique_ratio")), default=0.0) or 0.0
            dtype_family = str(prof.get("dtype_family", ""))

            # Simple scoring heuristic (v1)
            score = 0.0
            score += 0.65 * unique_ratio
            score += 0.25 * (1.0 - null_ratio)
            if dtype_family == "int":
                score += 0.10

            candidates.append(
                {
                    "columns": [col],
                    "score": clamp01(score),
                    "unique_ratio": unique_ratio,
                    "null_ratio": null_ratio,
                    "dtype_family": dtype_family,
                }
            )

    # Sort best-first
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # If none, return info record
    if not candidates:
        rec = RecommendationRecord(
            decision="PRIMARY_KEY",
            status="info",
            confidence=0.0,
            reason_codes=[],
            payload={"columns": [], "note": "No strong PK candidate found from unary UCC."},
            normalisation_disclosure=disclose_key_normalisation(trimmed=True, casefolded=True),
        )
        return rec.__dict__

    best = candidates[0]
    reason_codes = []
    if best["unique_ratio"] >= 0.999:
        reason_codes.append(R.PK_UNIQUE_HIGH)
    if best["null_ratio"] <= 0.01:
        reason_codes.append(R.PK_NULL_LOW)
    if best["dtype_family"] == "int":
        reason_codes.append(R.PK_DTYPE_NUMERIC)

    # referenced_by_fks evidence
    ref_count = 0
    if edges_df is not None and not edges_df.empty:
        if {"pk_table", "pk_column"}.issubset(edges_df.columns):
            ref_count = int(((edges_df["pk_table"] == table_name) & (edges_df["pk_column"] == best["columns"][0])).sum())
            if ref_count > 0:
                reason_codes.append(R.PK_REFERENCED_BY_FKS)

    rec = RecommendationRecord(
        decision="PRIMARY_KEY",
        status="recommended",
        confidence=float(best["score"]),
        reason_codes=reason_codes,
        evidence={
            "unique_ratio": best["unique_ratio"],
            "null_ratio": best["null_ratio"],
            "referenced_by_fk_edges": ref_count,
        },
        normalisation_disclosure=disclose_key_normalisation(trimmed=True, casefolded=True),
        payload={"columns": best["columns"]},
        alternatives=[],
    )

    # Build runner-ups
    alts: list[AltOption] = []
    for cand in top_n(candidates[1:], top_n_alternatives):
        alt_reasons: list[str] = []
        if cand["unique_ratio"] >= 0.999:
            alt_reasons.append(R.PK_UNIQUE_HIGH)
        if cand["null_ratio"] <= 0.01:
            alt_reasons.append(R.PK_NULL_LOW)
        if cand["dtype_family"] == "int":
            alt_reasons.append(R.PK_DTYPE_NUMERIC)

        alts.append(
            AltOption(
                payload={"columns": cand["columns"]},
                confidence=float(cand["score"]),
                reason_codes=alt_reasons,
                blockers=[],
                normalisation_disclosure=disclose_key_normalisation(trimmed=True, casefolded=True),
            )
        )

    # Attach alternatives
    rec = RecommendationRecord(
        **{**rec.__dict__, "alternatives": alts}  # type: ignore[arg-type]
    )
    return rec.__dict__
