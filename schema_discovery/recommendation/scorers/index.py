from __future__ import annotations

from typing import Any

import pandas as pd

from ..schema import RecommendationRecord
from ..utils import safe_int
from .. import reasons as R


def _profile_row(profiles_df: pd.DataFrame, table: str, col: str) -> dict[str, Any]:
    if profiles_df is None or profiles_df.empty:
        return {}
    sub = profiles_df[(profiles_df["table_name"] == table) & (profiles_df["column_name"] == col)]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


def recommend_indexes(
    *,
    table_name: str,
    pk_rec: dict[str, Any],
    fk_recs: list[dict[str, Any]],
    composite_recs: list[dict[str, Any]],
    profiles_df: pd.DataFrame,
) -> dict[str, Any]:
    indexes_recommended: list[dict[str, Any]] = []
    indexes_avoid: list[dict[str, Any]] = []

    # 1) PK -> index
    pk_cols = (pk_rec or {}).get("payload", {}).get("columns", [])
    if pk_cols:
        indexes_recommended.append(
            RecommendationRecord(
                decision="INDEX",
                status="recommended",
                confidence=0.95,
                reason_codes=[R.IDX_PRIMARY_KEY],
                payload={"columns": pk_cols, "priority": "HIGH", "name_suggestion": f"pk_{table_name}"},
            ).__dict__
        )

    # 2) FK columns -> index
    for fk in fk_recs or []:
        if fk.get("status") != "recommended":
            continue
        cols = fk.get("payload", {}).get("columns", [])
        if not cols:
            continue
        indexes_recommended.append(
            RecommendationRecord(
                decision="INDEX",
                status="recommended",
                confidence=float(fk.get("confidence", 0.8)),
                reason_codes=[R.IDX_FOREIGN_KEY],
                payload={
                    "columns": cols,
                    "priority": "HIGH",
                    "name_suggestion": f"idx_{table_name}_{cols[0]}",
                },
                evidence={"fk_confidence": fk.get("confidence")},
            ).__dict__
        )

    # 3) Composite keys -> composite index candidate
    for ck in composite_recs or []:
        cols = ck.get("payload", {}).get("columns", [])
        if not cols:
            continue
        indexes_recommended.append(
            RecommendationRecord(
                decision="INDEX",
                status="recommended",
                confidence=float(ck.get("confidence", 0.7)),
                reason_codes=[R.IDX_COMPOSITE_KEY],
                payload={
                    "columns": cols,
                    "priority": "MEDIUM",
                    "name_suggestion": f"idx_{table_name}_{'_'.join(cols)}",
                },
            ).__dict__
        )

    # 4) Avoid indexes on obvious bad columns (low cardinality or very long text)
    # This is conservative -> you can tighten later.
    # We only add avoids for columns that look clearly bad.
    if profiles_df is not None and not profiles_df.empty:
        tdf = profiles_df[profiles_df["table_name"] == table_name] if "table_name" in profiles_df.columns else pd.DataFrame()
        for r in tdf.to_dict(orient="records"):
            col = str(r.get("column_name"))
            max_len = safe_int(r.get("max_len"), default=0)
            n_unique = safe_int(r.get("n_unique"), default=0)
            n_rows = safe_int(r.get("n_rows"), default=0)

            if max_len >= 1000:
                indexes_avoid.append(
                    RecommendationRecord(
                        decision="INDEX",
                        status="info",
                        confidence=0.85,
                        reason_codes=[R.IDX_TEXT_AVOID],
                        payload={"columns": [col], "reason": "Very long text column"},
                        evidence={"max_len": max_len},
                    ).__dict__
                )
                continue

            if n_rows >= 1000:
                distinct_ratio = (n_unique / n_rows) if n_rows else 0.0
                if distinct_ratio <= 0.01:
                    indexes_avoid.append(
                        RecommendationRecord(
                            decision="INDEX",
                            status="info",
                            confidence=0.70,
                            reason_codes=[R.IDX_LOW_CARDINALITY_AVOID],
                            payload={
                                "columns": [col],
                                "reason": "Low selectivity -> avoid single-column index; consider composite index if commonly filtered",
                            },
                            evidence={"n_unique": n_unique, "n_rows": n_rows, "distinct_ratio": distinct_ratio},
                        ).__dict__
                    )


    return {"indexes_recommended": indexes_recommended, "indexes_avoid": indexes_avoid}
