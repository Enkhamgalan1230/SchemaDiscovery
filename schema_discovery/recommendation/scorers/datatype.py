from __future__ import annotations

from typing import Any

import pandas as pd

from ..disclosure import disclose_numeric_coercion, disclose
from ..schema import AltOption, Blocker, RecommendationRecord
from ..utils import clamp01, safe_float, safe_int, top_n
from .. import reasons as R


def _table_profiles(profiles_df: pd.DataFrame, table: str) -> pd.DataFrame:
    if profiles_df is None or profiles_df.empty:
        return pd.DataFrame()
    if "table_name" not in profiles_df.columns:
        return pd.DataFrame()
    return profiles_df[profiles_df["table_name"] == table].copy()


def _suggest_db_type(row: dict[str, Any]) -> tuple[str, float, list[str], list[AltOption], list[Blocker]]:
    """
    Conservative v1 mapping using common profiler fields.
    This is intentionally simple -> you can enrich using profile_extras later.
    """
    dtype_family = str(row.get("dtype_family", "")).lower()
    dtype = str(row.get("dtype", "")).lower()

    max_len = safe_int(row.get("max_len"), default=0)
    n_unique = safe_int(row.get("n_unique"), default=0)
    n_rows = safe_int(row.get("n_rows"), default=0)

    distinct_ratio = (float(n_unique) / float(n_rows)) if (n_rows > 0) else 0.0

    # numeric range if available
    vmin = row.get("min_value", row.get("min"))
    vmax = row.get("max_value", row.get("max"))
    fmin = safe_float(vmin, default=None)
    fmax = safe_float(vmax, default=None)

    reasons: list[str] = []
    blockers: list[Blocker] = []
    alts: list[AltOption] = []

    # UUID pattern hint often comes from enhanced profiler, but dtype may be "uuid"
    if "uuid" in dtype or "uuid" in dtype_family:
        reasons.append(R.DTYPE_UUID_REGEX)
        return ("UUID", 0.95, reasons, alts, blockers)

    if dtype_family == "bool":
        reasons.append(R.DTYPE_BOOL_PATTERN)
        return ("BOOLEAN", 0.90, reasons, alts, blockers)

    if dtype_family == "datetime" or "date" in dtype:
        reasons.append(R.DTYPE_DATE_PARSE_HIGH)
        return ("TIMESTAMP", 0.85, reasons, alts, blockers)

    if dtype_family == "int":
        # If range present, choose INT vs BIGINT vs SMALLINT
        if fmin is not None and fmax is not None:
            # safe bounds (approx) for SQL types
            if -32768 <= fmin and fmax <= 32767:
                reasons.append(R.DTYPE_INT_RANGE_SMALL)
                return ("SMALLINT", 0.85, reasons, alts, blockers)
            if -2147483648 <= fmin and fmax <= 2147483647:
                reasons.append(R.DTYPE_INT_RANGE_SMALL)
                return ("INT", 0.90, reasons, alts, blockers)

            reasons.append(R.DTYPE_INT_RANGE_BIG)
            # alternative INT is blocked
            alts.append(
                AltOption(
                    payload={"db_type": "INT"},
                    confidence=0.40,
                    reason_codes=[R.DTYPE_INT_RANGE_SMALL],
                    blockers=[
                        Blocker(
                            code=R.BLK_INT_OVERFLOW_RISK,
                            message="Observed range exceeds safe INT bounds.",
                            metrics={"min": fmin, "max": fmax},
                        )
                    ],
                    normalisation_disclosure=disclose_numeric_coercion(),
                )
            )
            return ("BIGINT", 0.92, reasons, alts, blockers)

        # no range -> default to BIGINT if it looks like an ID, else INT
        if "id" in str(row.get("column_name", "")).lower():
            reasons.append(R.DTYPE_INT_ID_HEURISTIC)
            return ("BIGINT", 0.75, reasons, alts, blockers)

        reasons.append(R.DTYPE_INT_NO_RANGE_DEFAULT)
        return ("INT", 0.70, reasons, alts, blockers)

    if dtype_family == "float":
        reasons.append(R.DTYPE_DECIMAL_LIKE)
        return ("DECIMAL(10,2)", 0.70, reasons, alts, blockers)

    # string-like
    if max_len >= 1000:
        reasons.append(R.DTYPE_TEXT_LONG)
        return ("TEXT", 0.85, reasons, alts, blockers)

    # small stable codes -> CHAR/VARCHAR
    if max_len > 0 and max_len <= 5 and distinct_ratio < 0.2:
        reasons.append(R.DTYPE_VARCHAR_FIT)
        return (f"CHAR({max_len})", 0.65, reasons, alts, blockers)

    if max_len > 0:
        reasons.append(R.DTYPE_VARCHAR_FIT)
        # pad a little to avoid tight fits
        n = max(1, min(255, max_len))
        return (f"VARCHAR({n})", 0.75, reasons, alts, blockers)

    # fallback
    return ("TEXT", 0.50, reasons, alts, blockers)

def _maybe_minmax_from_df(df: pd.DataFrame, col: str) -> tuple[float | None, float | None]:
    """
    If profiler didn't provide min/max, compute it from the actual dataframe.
    Safe: coerces to numeric, drops NaNs, returns (None, None) if empty/unavailable.
    """
    if df is None or df.empty or col not in df.columns:
        return (None, None)

    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return (None, None)

    return (float(s.min()), float(s.max()))

def recommend_datatypes(
    *,
    table_name: str,
    df: pd.DataFrame,  # <- add
    profiles_df: pd.DataFrame,
    profile_extras: dict[str, Any],
    top_n_alternatives: int,
) -> list[dict[str, Any]]:
    tdf = _table_profiles(profiles_df, table_name)
    if tdf.empty:
        return []

    out: list[dict[str, Any]] = []
    for r in tdf.to_dict(orient="records"):
        col = str(r.get("column_name"))

        # Pull profiler min/max (if any)
        vmin = r.get("min_value", r.get("min"))
        vmax = r.get("max_value", r.get("max"))

        # If missing, compute from actual df
        if vmin is None or vmax is None:
            mn, mx = _maybe_minmax_from_df(df, col)
            if vmin is None:
                vmin = mn
            if vmax is None:
                vmax = mx

        # Feed filled min/max into suggester
        db_type, conf, reasons, alts, blockers = _suggest_db_type(
            {**r, "column_name": col, "min_value": vmin, "max_value": vmax}
        )

        rec = RecommendationRecord(
            decision="DATATYPE",
            status="recommended",
            confidence=clamp01(conf),
            reason_codes=reasons,
            blockers=blockers,
            payload={"column": col, "db_type": db_type},
            alternatives=top_n(alts, top_n_alternatives),
            normalisation_disclosure=disclose_numeric_coercion() if "numeric" in " ".join(reasons) else disclose(),
            evidence={
                "dtype_family": r.get("dtype_family"),
                "dtype": r.get("dtype"),
                "max_len": r.get("max_len"),
                "min_value": vmin,  # <- use filled values
                "max_value": vmax,  # <- use filled values
            },
        )
        out.append(rec.__dict__)

    return out
