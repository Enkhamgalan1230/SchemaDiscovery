# schema_discovery/quality/key_representations.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KeyRepConfig:
    # Float or numeric token -> int string if "enough" values are integer-like.
    # This avoids failing the whole rep because of a few junk tokens.
    int_like_tol: float = 1e-9
    int_like_min_frac: float = 0.98  # require >= 98% integer-like among parseable numerics
    min_parseable_numeric_frac: float = 0.50  # if too few values parse as numeric -> skip numeric reps

    # Zfill variants
    enable_zfill: bool = True
    max_zfill_width: int = 18

    # Prevent silly numeric conversions
    max_abs_int: int = 2_147_483_647  # guard overflow-ish / very large IDs, adjust if needed


def _to_string(series: pd.Series) -> pd.Series:
    return series.astype("string")


def _strip(series: pd.Series) -> pd.Series:
    s = _to_string(series).str.strip()
    # Treat empty string as missing
    s = s.where(s.str.len() > 0)
    return s


def _numeric_coerce(series: pd.Series) -> pd.Series:
    """
    Convert tokens to numeric where possible. Returns float series with NaN for non-parseable.
    """
    s = series
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    tok = _to_string(s).str.strip()
    return pd.to_numeric(tok, errors="coerce")


def _num_int_str(series: pd.Series, cfg: KeyRepConfig) -> pd.Series:
    """
    Canonical integer string representation for integer-like numeric tokens.
    - Converts only if enough values are numeric and integer-like.
    - Non-parseable or non-integer-like tokens become <NA>.
    """
    s = _strip(series)
    num = _numeric_coerce(s)

    out = pd.Series(pd.array([pd.NA] * len(s), dtype="string"), index=s.index)

    parseable = num.notna()
    if not parseable.any():
        return out

    parseable_frac = float(parseable.mean())
    if parseable_frac < float(cfg.min_parseable_numeric_frac):
        return out

    vals = num.loc[parseable].to_numpy(dtype=np.float64, copy=False)
    int_like = np.isclose(vals, np.round(vals), atol=float(cfg.int_like_tol))
    int_like_frac = float(int_like.mean()) if len(int_like) else 0.0

    if int_like_frac < float(cfg.int_like_min_frac):
        # too many fractional numbers -> not a safe integer-key rep
        return out

    # Only emit for int-like values. Others -> NA.
    ok_idx = num.loc[parseable].index[int_like]
    ints = np.round(num.loc[ok_idx].to_numpy(dtype=np.float64, copy=False)).astype(np.int64)

    # guard huge values (optional but helps avoid "1e20" nonsense becoming int)
    if cfg.max_abs_int is not None:
        mask_ok = np.abs(ints) <= int(cfg.max_abs_int)
        ok_idx = ok_idx[mask_ok]
        ints = ints[mask_ok]

    out.loc[ok_idx] = pd.Series(ints.astype(str), index=ok_idx, dtype="string")
    return out


def _infer_zfill_width(a: pd.Series, b: pd.Series, max_width: int) -> Optional[int]:
    """
    Infer a reasonable padding width from digit-only tokens across both series.
    Uses max digit length observed, capped.
    """
    def _max_len_digits(s: pd.Series) -> int:
        s = _strip(s).dropna()
        if s.empty:
            return 0
        d = s[s.str.fullmatch(r"\d+")]
        if d.empty:
            return 0
        return int(d.str.len().max())

    w = max(_max_len_digits(a), _max_len_digits(b))
    if w <= 1:
        return None
    if w > int(max_width):
        return None
    return int(w)


def _zfill(series: pd.Series, width: int) -> pd.Series:
    s = _strip(series)
    out = s.copy()
    mask = s.notna() & s.str.fullmatch(r"\d+")
    out.loc[mask] = s.loc[mask].str.zfill(int(width))
    return out

def _digits_lstrip_zeros(series: pd.Series) -> pd.Series:
    """
    For digit-only tokens, remove leading zeros.
    Keeps "0" as "0".
    Non-digit tokens -> <NA>.
    """
    s = _strip(series)
    out = pd.Series(pd.array([pd.NA] * len(s), dtype="string"), index=s.index)

    mask = s.notna() & s.str.fullmatch(r"\d+")
    if not mask.any():
        return out

    vals = s.loc[mask].astype("string")
    stripped = vals.str.lstrip("0")
    stripped = stripped.where(stripped.str.len() > 0, "0")

    out.loc[mask] = stripped
    return out

def build_key_representations(
    series: pd.Series,
    *,
    other: Optional[pd.Series] = None,
    cfg: Optional[KeyRepConfig] = None,
) -> Dict[str, pd.Series]:
    """
    Named representations used by soft IND matching.
    Keep this small and deliberate to avoid combinatorial explosion.
    """
    cfg = cfg or KeyRepConfig()

    reps: Dict[str, pd.Series] = {}
    reps["raw_str"] = _strip(series)

    reps["num_int_str"] = _num_int_str(series, cfg)

    # Handles 0000000775 vs 775 style joins
    reps["digits_lstrip_zeros"] = _digits_lstrip_zeros(series)

    if cfg.enable_zfill and other is not None:
        width = _infer_zfill_width(series, other, cfg.max_zfill_width)
        if width is not None:
            reps[f"zfill_{width}"] = _zfill(series, width)
            reps[f"num_int_str_zfill_{width}"] = _zfill(reps["num_int_str"], width)

    return reps
