from __future__ import annotations

from typing import Any

import pandas as pd

from ..schema import AltOption, Blocker, RecommendationRecord
from ..utils import clamp01, safe_float, top_n
from .. import reasons as R


def _match_rel_missing(rel_missing_df: pd.DataFrame, edge: dict[str, Any]) -> dict[str, Any] | None:
    """
    Try to match rel_missing rows to an edge. Your rel_missing schema may evolve,
    so this function is intentionally defensive.
    """
    if rel_missing_df is None or rel_missing_df.empty:
        return None

    cols = set(rel_missing_df.columns)
    needed = {"fk_table", "fk_column", "pk_table", "pk_column"}
    if not needed.issubset(cols):
        return None

    sub = rel_missing_df[
        (rel_missing_df["fk_table"] == edge.get("fk_table"))
        & (rel_missing_df["fk_column"] == edge.get("fk_column"))
        & (rel_missing_df["pk_table"] == edge.get("pk_table"))
        & (rel_missing_df["pk_column"] == edge.get("pk_column"))
    ]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()


def recommend_foreign_keys(
    *,
    table_name: str,
    edges_df: pd.DataFrame,
    rel_missing_df: pd.DataFrame,
    default_fk_df: pd.DataFrame,
    top_n_alternatives: int,
) -> list[dict[str, Any]]:
    if edges_df is None or edges_df.empty:
        return []

    # edges touching this table as child
    if not {"fk_table", "fk_column", "pk_table", "pk_column"}.issubset(edges_df.columns):
        return []

    sub = edges_df[edges_df["fk_table"] == table_name]
    if sub.empty:
        return []

    rows = sub.to_dict(orient="records")

    recs: list[dict[str, Any]] = []
    for edge in rows:
        score = safe_float(edge.get("score"), default=0.0) or 0.0
        distinct_cov = safe_float(edge.get("distinct_coverage"), default=None)
        row_cov = safe_float(edge.get("row_coverage"), default=None)

        reason_codes = []
        if score >= 0.90:
            reason_codes.append(R.FK_SCORE_HIGH)
        if distinct_cov is not None and distinct_cov >= 0.90:
            reason_codes.append(R.FK_INCLUSION_HIGH)

        status: str = "recommended"
        blockers: list[Blocker] = []

        rm = _match_rel_missing(rel_missing_df, edge)
        if rm:
            # Try common field names for missing parent ratio
            missing_parent_ratio = None
            for k in ("orphan_ratio", "orphan_fk_ratio", "missing_parent_ratio", "fk_orphan_ratio"):
                if k in rm:
                    missing_parent_ratio = safe_float(rm.get(k), default=None)
                    break

            if missing_parent_ratio is not None:
                # v1 threshold: block if too high
                if missing_parent_ratio >= 0.05:
                    status = "blocked"
                    blockers.append(
                        Blocker(
                            code=R.BLK_FK_MISSING_PARENT_HIGH,
                            message="Too many FK values do not exist in the parent domain.",
                            metrics={"missing_parent_ratio": missing_parent_ratio},
                        )
                    )
                else:
                    reason_codes.append(R.FK_MISSING_PARENT_LOW)

        rec = RecommendationRecord(
            decision="FOREIGN_KEY",
            status=status,  # type: ignore[arg-type]
            confidence=clamp01(score),
            reason_codes=reason_codes,
            blockers=blockers,
            evidence={
                "score": score,
                **({"distinct_coverage": distinct_cov} if distinct_cov is not None else {}),
                **({"row_coverage": row_cov} if row_cov is not None else {}),
            },
            payload={
                "columns": [edge["fk_column"]],
                "references": {"table": edge["pk_table"], "columns": [edge["pk_column"]]},
            },
            alternatives=[],
        )

        recs.append(rec.__dict__)

    # For now, no runner-ups per FK column. You can add later by grouping per (fk_table,fk_column).
    return recs
