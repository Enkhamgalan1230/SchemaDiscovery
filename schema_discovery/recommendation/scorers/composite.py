from __future__ import annotations

from typing import Any

import pandas as pd

from ..schema import AltOption, RecommendationRecord
from ..utils import clamp01, safe_float, top_n
from .. import reasons as R


def recommend_composite_keys(
    *,
    table_name: str,
    composite_ucc_df: pd.DataFrame,
    top_n_alternatives: int,
) -> list[dict[str, Any]]:
    if composite_ucc_df is None or composite_ucc_df.empty:
        return []

    # Expect columns like: table_name, columns (list) OR col_1..col_k, unique_ratio, null_ratio
    sub = composite_ucc_df[composite_ucc_df["table_name"] == table_name] if "table_name" in composite_ucc_df.columns else pd.DataFrame()
    if sub.empty:
        return []

    rows = sub.to_dict(orient="records")
    scored: list[dict[str, Any]] = []

    for r in rows:
        cols = r.get("columns")
        if cols is None:
            # fallback: collect any keys like col_1, col_2...
            cols = [r[k] for k in sorted(r.keys()) if str(k).startswith("col_") and r.get(k)]
        cols = [str(c) for c in (cols or [])]
        if not cols:
            continue

        unique_ratio = safe_float(r.get("unique_ratio"), default=0.0) or 0.0
        null_ratio = safe_float(r.get("null_ratio"), default=0.0) or 0.0

        score = 0.7 * unique_ratio + 0.3 * (1.0 - null_ratio)
        scored.append({"columns": cols, "score": clamp01(score), "unique_ratio": unique_ratio, "null_ratio": null_ratio})

    scored.sort(key=lambda x: x["score"], reverse=True)
    if not scored:
        return []

    out: list[dict[str, Any]] = []
    for best in top_n(scored, top_n_alternatives + 1):
        rec = RecommendationRecord(
            decision="COMPOSITE_KEY",
            status="recommended",
            confidence=float(best["score"]),
            reason_codes=[R.COMPOSITE_UNIQUE_HIGH] if best["unique_ratio"] >= 0.999 else [],
            evidence={"unique_ratio": best["unique_ratio"], "null_ratio": best["null_ratio"]},
            payload={"columns": best["columns"]},
            alternatives=[],
        )
        out.append(rec.__dict__)

    return out
