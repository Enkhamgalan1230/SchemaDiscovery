"""
Unified profiler for schema discovery.

This replaces:
- column_profiler.py (basic)
- enhanced_profiler.py (enhanced)

Contract (must stay valid):
- discover_ucc_unary(profiles_df) requires:
  table_name, column_name, dtype_family, n_rows, unique_ratio, null_ratio (+ optional is_unary_ucc)
- discover_ind_unary(..., profiles_df) requires:
  table_name, column_name, dtype_family, null_ratio, n_rows, n_unique, unique_ratio

Design:
- mode="basic" -> compute only required fields cheaply (pandas dtype families)
- mode="enhanced" -> still compute required fields, but use AtomicDtypeModelV2 for dtype_family
  and compute extra metrics (quality, numeric/date stats, etc)
- Never mutates dfs
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional, Literal

import numpy as np
import pandas as pd

from schema_discovery.profiling.atomic_model import AtomicDtypeModelV2, InferenceConfig, AtomicDType
from schema_discovery.quality.key_normalisation import normalise_null_like

ProfileMode = Literal["basic", "enhanced"]


# -----------------------------
# Helpers (shared)
# -----------------------------
def _dtype_family_from_pandas(dtype: Any) -> str:
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    if pd.api.types.is_integer_dtype(dtype):
        return "int"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        return "string"
    return "other"

def _top1_ratio(s: pd.Series) -> float:
    """
    Ratio of the most frequent value among non-null values.
    0.0 means empty or no non-null values.
    """
    vals = s.dropna()
    if vals.empty:
        return 0.0
    vc = vals.value_counts(normalize=True, dropna=True)
    if vc.empty:
        return 0.0
    return float(vc.iloc[0])


def _dtype_family_from_atomic(at: AtomicDType) -> str:
    if at is AtomicDType.BOOL:
        return "bool"
    if at is AtomicDType.INT:
        return "int"
    if at is AtomicDType.FLOAT:
        return "float"
    if at is AtomicDType.DATE:
        return "datetime"
    return "string"


def _safe_nunique(s: pd.Series) -> int:
    return int(s.nunique(dropna=True))


def _sample_values(s: pd.Series, k: int = 5) -> list[str]:
    if k <= 0:
        return []
    vals = s.dropna()
    if vals.empty:
        return []
    uniq = vals.drop_duplicates().head(k)
    return [str(x) for x in uniq.tolist()]


def _string_length_stats(s: pd.Series) -> dict[str, float | None]:
    vals = s.dropna()
    if vals.empty:
        return {"avg_len": None, "min_len": None, "max_len": None}
    lens = vals.astype(str).map(len)
    return {
        "avg_len": float(lens.mean()),
        "min_len": float(lens.min()),
        "max_len": float(lens.max()),
    }

def _health_check(
    *,
    dtype_family: str,
    n_rows: int,
    n_null: int,
    n_unique: int,
    unique_ratio_non_null: float,
    avg_len: float | None,
    min_len: float | None,
    max_len: float | None,
) -> tuple[str, str, list[str], int]:
    """
    Return (status, severity, reasons, score).
    score is 0..100 where 100 is best.
    """
    if n_rows <= 0:
        return "fail", "high", ["empty_table_or_column"], 0

    null_ratio = (n_null / n_rows) if n_rows else 1.0
    non_null = n_rows - n_null

    reasons: list[str] = []
    score = 100

    # Nullness signals
    if null_ratio >= 0.95:
        reasons.append("mostly_null")
        score -= 50
    elif null_ratio >= 0.50:
        reasons.append("high_null_ratio")
        score -= 25
    elif null_ratio >= 0.20:
        reasons.append("moderate_null_ratio")
        score -= 10

    # Low information columns
    if non_null > 0 and n_unique <= 1:
        reasons.append("constant_or_near_constant")
        score -= 35

    # String hygiene
    if dtype_family == "string" and non_null > 0:
        if avg_len is not None and avg_len == 0:
            reasons.append("empty_strings_dominate")
            score -= 25

        # suspicious if huge length spread (often free text mixed with codes)
        if min_len is not None and max_len is not None and min_len >= 0 and max_len >= 0:
            if max_len - min_len >= 50:
                reasons.append("high_length_variance")
                score -= 10

    # Key-ish signal: very high uniqueness but with nulls
    # this often means "id column with missing values" -> warn
    if non_null > 0 and unique_ratio_non_null >= 0.99 and n_null > 0:
        reasons.append("near_unique_but_has_nulls")
        score -= 10

    score = int(max(0, min(100, score)))

    # Map to status / severity
    if score >= 85:
        return "ok", "low", reasons, score

    if score >= 70:
        return "warn", "low", reasons, score

    if score >= 40:
        return "warn", "medium", reasons, score

    return "fail", "high", reasons, score


# -----------------------------
# Enhanced-only helpers
# -----------------------------
def _pk_continuity_int(series_numeric: pd.Series) -> float | None:
    valid = series_numeric.dropna()
    if valid.empty:
        return None
    try:
        arr = np.sort(pd.unique(valid))
        if arr.size < 2:
            return 100.0
        min_val, max_val = int(arr[0]), int(arr[-1])
        expected = (max_val - min_val + 1)
        if expected <= 0:
            return None
        missing = expected - int(arr.size)
        continuity = 1.0 - (missing / expected)
        return float(np.clip(continuity * 100.0, 0.0, 100.0))
    except Exception:
        return None


def _freshness_score(max_date: pd.Timestamp) -> float | None:
    try:
        now = pd.Timestamp.now(tz=max_date.tz) if max_date.tzinfo else pd.Timestamp.now()
        age_days = (now - max_date).days
        score = 100.0 - (age_days / 365.0) * 100.0
        return float(np.clip(score, 0.0, 100.0))
    except Exception:
        return None


def _consistency_score(series: pd.Series, dtype_family: str) -> float:
    valid = series.dropna()
    if valid.empty:
        return 0.0

    try:
        if dtype_family in {"int", "float"}:
            x = pd.to_numeric(valid, errors="coerce").dropna()
            if x.empty:
                return 40.0
            mean = float(x.mean())
            std = float(x.std()) if len(x) > 1 else 0.0
            if mean == 0.0:
                return 90.0
            cv = abs(std / mean) if mean != 0 else 0.0
            return float(np.clip((1.0 - min(cv, 1.0)) * 100.0, 0.0, 100.0))

        if dtype_family == "string":
            s = valid.astype(str)
            lengths = s.str.len()
            mean_len = float(lengths.mean()) if len(lengths) else 0.0
            std_len = float(lengths.std()) if len(lengths) > 1 else 0.0
            length_cv = (std_len / mean_len) if mean_len > 0 else 0.0
            length_score = float(np.clip((1.0 - min(length_cv, 1.0)) * 100.0, 0.0, 100.0))

            sample = s.head(min(200, len(s)))
            patt = []
            for v in sample:
                if not v:
                    continue
                a = sum(c.isalpha() for c in v) / len(v)
                d = sum(c.isdigit() for c in v) / len(v)
                patt.append((a, d))
            if not patt:
                return length_score

            a_std = float(np.std([p[0] for p in patt]))
            d_std = float(np.std([p[1] for p in patt]))
            pattern_score = float(np.clip((1.0 - min((a_std + d_std) / 2.0, 1.0)) * 100.0, 0.0, 100.0))
            return float((length_score + pattern_score) / 2.0)

        if dtype_family == "datetime":
            dt = pd.to_datetime(valid, errors="coerce").dropna()
            if len(dt) < 2:
                return 100.0
            dt_sorted = dt.sort_values()
            diffs = dt_sorted.diff().dropna()
            if diffs.empty:
                return 100.0
            secs = diffs.dt.total_seconds()
            mean = float(secs.mean())
            std = float(secs.std()) if len(secs) > 1 else 0.0
            if mean == 0.0:
                return 90.0
            cv = abs(std / mean) if mean != 0 else 0.0
            return float(np.clip((1.0 - min(cv, 1.0)) * 100.0, 0.0, 100.0))

        if dtype_family == "bool":
            vc = valid.value_counts(normalize=True)
            if len(vc) < 2:
                return 20.0
            minority = float(min(vc.values))
            return float(np.clip(minority * 200.0, 0.0, 100.0))

        return 70.0
    except Exception:
        return 50.0


def _numeric_stats(series: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std_dev": None,
            "q25": None,
            "q75": None,
            "iqr": None,
            "skewness": None,
        }

    q25 = x.quantile(0.25)
    q75 = x.quantile(0.75)
    return {
        "mean": float(x.mean()),
        "median": float(x.median()),
        "min": float(x.min()),
        "max": float(x.max()),
        "std_dev": float(x.std()) if len(x) > 1 else 0.0,
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "skewness": float(x.skew()) if len(x) > 2 else 0.0,
    }


def _date_stats(series: pd.Series) -> dict[str, Any]:
    dt = pd.to_datetime(series, errors="coerce").dropna()
    if dt.empty:
        return {
            "date_min": None,
            "date_max": None,
            "date_range_days": None,
            "freshness_score": None,
        }
    dt_sorted = dt.sort_values()
    mn = dt_sorted.min()
    mx = dt_sorted.max()
    rng = (mx - mn).days
    return {
        "date_min": mn.isoformat(),
        "date_max": mx.isoformat(),
        "date_range_days": int(rng),
        "freshness_score": _freshness_score(mx),
    }


# -----------------------------
# Public API
# -----------------------------
def profile_table(
    df: pd.DataFrame,
    table_name: str,
    *,
    sample_k: int = 5,
    mode: ProfileMode = "basic",
    atomic_cfg: Optional[InferenceConfig] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Profile a single table.

    Returns:
    -> profiles_df: one row per column (always includes required base columns)
    -> extras: table-level metadata (empty in basic mode)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    mode = str(mode).lower()
    if mode not in ("basic", "enhanced"):
        raise ValueError("mode must be 'basic' or 'enhanced'")

    n_rows = int(len(df))

    # Only fit atomic model in enhanced mode (expensive part)
    schema = None
    cfg = None
    if mode == "enhanced":
        cfg = atomic_cfg or InferenceConfig(profile=False)
        atomic_model = AtomicDtypeModelV2(cfg).fit(df)
        schema = atomic_model.schema()  # name -> ColumnInference

    rows: list[dict[str, Any]] = []
    extras_columns: dict[str, Any] = {}

    for col in df.columns:
        s_raw = df[col]
        col_name = str(col)

        # ---- base stats (always computed) ----
        s_clean = normalise_null_like(s_raw)

        n_null = int(s_clean.isna().sum())
        n_non_null = int(n_rows - n_null)
        null_ratio = (n_null / n_rows) if n_rows else 0.0

        # Uniqueness should be computed over non-null values
        s_non_null = s_clean.dropna()
        n_unique = int(s_non_null.nunique(dropna=True))

        top1_ratio = _top1_ratio(s_clean)

        unique_ratio = (n_unique / n_rows) if n_rows else 0.0
        unique_ratio_non_null = (n_unique / n_non_null) if n_non_null else 0.0

        is_unary_ucc = (n_rows > 0) and (n_null == 0) and (n_unique == n_rows)


        # dtype_family decision depends on mode
        if mode == "enhanced" and schema is not None:
            inf = schema.get(col_name)
            if inf is None:
                atomic_type = AtomicDType.STRING
                atomic_prob = 0.0
                atomic_reason = "Missing inference result"
                convertible_ratio = 0.0
            else:
                atomic_type = inf.atomic_type
                atomic_prob = float(inf.confidence)
                atomic_reason = str(inf.reason)
                convertible_ratio = float(inf.convertible_ratio)

            dtype_family = _dtype_family_from_atomic(atomic_type)
        else:
            atomic_type = None
            atomic_prob = None
            atomic_reason = None
            convertible_ratio = None
            dtype_family = _dtype_family_from_pandas(s_raw.dtype)

        # length stats only meaningful for string family
        if dtype_family == "string":
            len_stats = _string_length_stats(s_clean)
        else:
            len_stats = {"avg_len": None, "min_len": None, "max_len": None}

        health_status, health_severity, health_reasons, health_score = _health_check(
            dtype_family=dtype_family,
            n_rows=n_rows,
            n_null=n_null,
            n_unique=n_unique,
            unique_ratio_non_null=float(unique_ratio_non_null),
            avg_len=len_stats["avg_len"],
            min_len=len_stats["min_len"],
            max_len=len_stats["max_len"],
        )


        row: dict[str, Any] = {
            # Required base columns (match your old profiler schema)
            "table_name": str(table_name),
            "column_name": col_name,
            "column_key": f"{table_name}.{col_name}",
            "dtype": str(s_raw.dtype),
            "dtype_family": dtype_family,
            "n_rows": n_rows,
            "n_null": n_null,
            "n_non_null": n_non_null,
            "null_ratio": float(null_ratio),
            "n_unique": int(n_unique),
            "unique_ratio": float(unique_ratio),
            "unique_ratio_non_null": float(unique_ratio_non_null),
            "is_unary_ucc": bool(is_unary_ucc),
            "sample_values": _sample_values(s_clean, k=sample_k),
            "avg_len": len_stats["avg_len"],
            "min_len": len_stats["min_len"],
            "max_len": len_stats["max_len"],
            "health_status": health_status,
            "health_severity": health_severity,
            "health_score": int(health_score),
            "health_reasons": health_reasons,
            "top1_ratio": float(top1_ratio),

        }

        # ---- enhanced extras (only if mode == enhanced) ----
        if mode == "enhanced":
            completeness = float((n_non_null / n_rows) * 100.0) if n_rows else 100.0
            consistency = _consistency_score(s_clean, dtype_family)
            uniqueness_pct = float(np.clip(unique_ratio * 100.0, 0.0, 100.0))
            overall_quality = float(np.clip(float(np.mean([completeness, uniqueness_pct, consistency])), 0.0, 100.0))

            pk_cont = None
            num_stats: dict[str, Any] = {}
            date_info: dict[str, Any] = {}

            if dtype_family == "int":
                x = pd.to_numeric(s_clean, errors="coerce")
                pk_cont = _pk_continuity_int(x)
                num_stats = _numeric_stats(s_clean)
            elif dtype_family == "float":
                num_stats = _numeric_stats(s_clean)
            elif dtype_family == "datetime":
                date_info = _date_stats(s_clean)

            row.update(
                {
                    "atomic_type": atomic_type.value if atomic_type is not None else None,
                    "atomic_prob": atomic_prob,
                    "atomic_convertible_ratio": convertible_ratio,
                    "atomic_reason": atomic_reason,
                    "completeness": completeness,
                    "consistency_score": consistency,
                    "overall_quality": overall_quality,
                    "pk_continuity": pk_cont,
                }
            )

            for k, v in num_stats.items():
                row[f"num_{k}"] = v

            for k, v in date_info.items():
                row[k] = v

            extras_columns[col_name] = {
                "dtype_family": dtype_family,
                "atomic_type": atomic_type.value if atomic_type is not None else "string",
                "atomic_prob": float(atomic_prob) if atomic_prob is not None else 0.0,
                "atomic_reason": atomic_reason or "",
                "quality": {
                    "completeness": completeness,
                    "consistency_score": consistency,
                    "overall_quality": overall_quality,
                },
            }

        rows.append(row)

    profiles_df = (
        pd.DataFrame(rows)
        .sort_values(["table_name", "column_name"], kind="mergesort")
        .reset_index(drop=True)
    )

    health_counts = profiles_df["health_status"].value_counts(dropna=False).to_dict()


    extras: dict[str, Any] = {}
    if mode == "enhanced":
        extras = {
            "table_name": str(table_name),
            "n_rows": n_rows,
            "n_cols": int(df.shape[1]),
            "atomic_config": asdict(cfg) if cfg is not None else {},
            "columns": extras_columns,
        }

    # Always attach health (cheap + useful)
    extras["health"] = {
        "counts": health_counts,
        "avg_score": float(profiles_df["health_score"].mean()) if "health_score" in profiles_df.columns else None,
    }
    return profiles_df, extras


def profile_all_tables(
    dfs: dict[str, pd.DataFrame],
    *,
    sample_k: int = 5,
    mode: ProfileMode = "basic",
    atomic_cfg: Optional[InferenceConfig] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not isinstance(dfs, dict) or not dfs:
        raise ValueError("dfs must be a non-empty dict[str, DataFrame]")

    frames: list[pd.DataFrame] = []
    extras: dict[str, Any] = {"tables": {}}  # ALWAYS

    for table_name, df in dfs.items():
        t_profiles, t_extras = profile_table(
            df,
            table_name=str(table_name),
            sample_k=sample_k,
            mode=mode,
            atomic_cfg=atomic_cfg,
        )
        frames.append(t_profiles)
        extras["tables"][str(table_name)] = t_extras  # ALWAYS

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = out.sort_values(["table_name", "column_name"], kind="mergesort").reset_index(drop=True)

    # Optional: global health rollup across all tables
    try:
        scores = []
        for tname, tex in extras["tables"].items():
            h = tex.get("health", {})
            if isinstance(h, dict) and h.get("avg_score") is not None:
                scores.append(float(h["avg_score"]))
        extras["health"] = {"avg_score": float(np.mean(scores)) if scores else None}
    except Exception:
        extras["health"] = {"avg_score": None}

    return out, extras

