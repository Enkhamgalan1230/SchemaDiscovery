from __future__ import annotations

import re
import numpy as np
import pandas as pd


# Keep this list deliberate. If you add too many, you'll erase legitimate codes.
DEFAULT_NULL_TOKENS: set[str] = {
    "",  # after strip
     "n/a", "n\\a",
    "nan", "-nan",
    "null", "none", "nil",
    "<na>", "<n/a>",
    "#n/a", "#n/a n/a", "#na",
    "-1.#ind", "1.#ind",
    "-1.#qnan", "1.#qnan",
}


_ws_re = re.compile(r"\s+")


def normalise_null_like(
    s: pd.Series,
    *,
    null_tokens: set[str] | None = None,
    lowercase: bool = True,
    collapse_ws: bool = True,
    strip: bool = True,
) -> pd.Series:
    """
    Convert common string placeholders for missing values into real missing (pd.NA).

    This does NOT drop rows. It only replaces values with pd.NA.
    """
    if null_tokens is None:
        null_tokens = DEFAULT_NULL_TOKENS

    out = s.copy()

    # Only touch non-null values
    mask = ~pd.isna(out)
    if not mask.any():
        return out

    tmp = out.loc[mask].astype(str)

    if collapse_ws:
        tmp = tmp.map(lambda x: _ws_re.sub(" ", x))
    if strip:
        tmp = tmp.str.strip()
    if lowercase:
        tmp_cmp = tmp.str.lower()
    else:
        tmp_cmp = tmp

    is_null = tmp_cmp.isin(null_tokens)
    if is_null.any():
        out.loc[tmp.index[is_null]] = pd.NA

    return out


def norm_key_series(
    s: pd.Series,
    *,
    normalise_string_nulls: bool = True,
    null_tokens: set[str] | None = None,
) -> pd.Series:
    """
    Normalise key columns so FK/PK comparisons are stable.

    - Optionally converts string placeholders for nulls -> pd.NA
    - Drops nulls (real + converted)
    - Normalises integer-like floats (999999.0 -> 999999)
    - Preserves non-integer numeric keys
    - Falls back to string keys for mixed / object columns
    """
    if normalise_string_nulls:
        s = normalise_null_like(s, null_tokens=null_tokens)

    s = s.dropna()

    if s.empty:
        # Keep consistent dtype for downstream comparisons
        return s.astype(object)

    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce").dropna()
        if x.empty:
            return s.astype("string").str.strip().astype(object)

        # All integer-like -> cast safely
        arr = x.to_numpy()
        if np.isclose(arr, np.round(arr)).all():
            return x.round().astype("Int64").astype(object)

        return x.astype(object)

    # object / string-like
    # strip is still useful even if normalise_string_nulls=False
    return s.astype("string").str.strip().astype(object)