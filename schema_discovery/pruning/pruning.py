# schema_discovery/pruning.py

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Optional

import pandas as pd


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class PruningConfig:
    # hard rejects
    reject_null_ratio: float = 0.98
    reject_top1_ratio: float = 0.999

    reject_boolean: bool = True
    reject_free_text: bool = True
    reject_measure_like: bool = True
    reject_date_like_name: bool = True
    reject_encoded_payload: bool = True

    # soft flags
    small_domain_distinct_max: int = 50

    # long text guard
    reject_long_text_avg_len: float = 120.0
    reject_long_text_max_len: int = 250

    # token sets for exact token matching, not substring matching
    key_tokens: tuple[str, ...] = (
        "id", "key", "code", "ref", "seq", "no", "num", "uuid", "guid"
    )

    measure_tokens: tuple[str, ...] = (
        "amount", "price", "total", "qty", "quantity", "score", "rating",
        "count", "cost", "minute", "minutes", "hour", "hours",
        "day", "days", "week", "weeks", "month", "months", "year", "years"
    )

    date_tokens: tuple[str, ...] = (
        "date", "time", "timestamp", "created", "updated", "modified"
    )

    negative_tokens: tuple[str, ...] = (
        "description", "comment", "comments", "note", "notes", "remark", "remarks",
        "message", "summary", "detail", "details", "content", "body", "text",
        "html", "xml", "json", "payload", "blob", "image", "file", "attachment"
    )

    payload_tokens: tuple[str, ...] = (
        "payload", "blob", "image", "file", "document", "content", "body", "attachment"
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
_CAMEL_RE_1 = re.compile(r"([a-z0-9])([A-Z])")
_CAMEL_RE_2 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        if v is None or pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _safe_int(v: object, default: int = 0) -> int:
    try:
        if v is None or pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


def split_name_tokens(name: str) -> list[str]:
    """
    Split column names into tokens using:
    - camel case
    - underscores
    - spaces
    - punctuation

    Examples:
    AccountCode -> ["account", "code"]
    HHldSeqNo -> ["h", "hld", "seq", "no"]   # imperfect but useful
    client_id -> ["client", "id"]
    Number/Email -> ["number", "email"]
    """
    if not isinstance(name, str) or not name.strip():
        return []

    s = name.strip()
    s = _CAMEL_RE_2.sub(r"\1 \2", s)
    s = _CAMEL_RE_1.sub(r"\1 \2", s)
    parts = _SPLIT_RE.split(s)

    out: list[str] = []
    for p in parts:
        p = p.strip().lower()
        if p:
            out.append(p)
    return out


def _has_any_token(tokens: Iterable[str], vocab: Iterable[str]) -> bool:
    token_set = set(tokens)
    vocab_set = set(vocab)
    return bool(token_set & vocab_set)


def _looks_encoded_payload(
    column_name: str,
    tokens: list[str],
    avg_len: float,
    max_len: int,
    unique_ratio_non_null: float,
    key_name_hint: bool,
    cfg: PruningConfig,
) -> bool:
    """
    Reject payload-like encoded columns, not business identifiers.
    """
    payload_name = _has_any_token(tokens, cfg.payload_tokens)
    long_textish = (avg_len >= 80.0) or (max_len >= 200)
    highly_unique = unique_ratio_non_null >= 0.90

    return payload_name and long_textish and highly_unique and (not key_name_hint)


# ---------------------------------------------------------------------
# Main pruning API
# ---------------------------------------------------------------------
def prune_profiles(
    profiles_df: pd.DataFrame,
    cfg: Optional[PruningConfig] = None,
) -> pd.DataFrame:
    """
    Coarse pruning layer.

    Goal:
    - reject obvious junk columns
    - keep lightweight flags and reasons
    - do NOT decide strict vs soft parent/child roles
    """
    cfg = cfg or PruningConfig()

    required = {
        "table_name",
        "column_name",
        "dtype_family",
        "null_ratio",
        "top1_ratio",
        "n_unique",
        "n_non_null",
        "unique_ratio_non_null",
        "avg_len",
        "max_len",
    }
    missing = required - set(profiles_df.columns)
    if missing:
        raise ValueError(f"profiles_df missing required columns for pruning: {sorted(missing)}")

    rows: list[dict[str, object]] = []

    for _, row in profiles_df.iterrows():
        table_name = str(row["table_name"])
        column_name = str(row["column_name"])
        dtype_family = str(row["dtype_family"])

        tokens = split_name_tokens(column_name)

        null_ratio = _safe_float(row.get("null_ratio"), 0.0)
        top1_ratio = _safe_float(row.get("top1_ratio"), 0.0)
        n_unique = _safe_int(row.get("n_unique"), 0)
        n_non_null = _safe_int(row.get("n_non_null"), 0)
        unique_ratio_non_null = _safe_float(row.get("unique_ratio_non_null"), 0.0)
        avg_len = _safe_float(row.get("avg_len"), 0.0)
        max_len = _safe_int(row.get("max_len"), 0)

        key_name_hint = _has_any_token(tokens, cfg.key_tokens)
        looks_measure_like = _has_any_token(tokens, cfg.measure_tokens)
        looks_date_like_name = _has_any_token(tokens, cfg.date_tokens)
        looks_negative_name = _has_any_token(tokens, cfg.negative_tokens)

        small_domain = (n_unique > 0) and (n_unique <= int(cfg.small_domain_distinct_max))

        looks_free_text = (
            dtype_family == "string"
            and (
                avg_len >= float(cfg.reject_long_text_avg_len)
                or max_len >= int(cfg.reject_long_text_max_len)
                or (avg_len >= 60.0 and max_len >= 1000)
            )
        )

        looks_encoded_payload = _looks_encoded_payload(
            column_name=column_name,
            tokens=tokens,
            avg_len=avg_len,
            max_len=max_len,
            unique_ratio_non_null=unique_ratio_non_null,
            key_name_hint=key_name_hint,
            cfg=cfg,
        )

        reject = False
        reasons: list[str] = []

        # hard rejects
        if n_non_null == 0:
            reject = True
            reasons.append("all_null")

        if null_ratio >= float(cfg.reject_null_ratio):
            reject = True
            reasons.append("too_many_nulls")

        if top1_ratio >= float(cfg.reject_top1_ratio):
            reject = True
            reasons.append("single_value_dominated")

        if cfg.reject_boolean and dtype_family == "bool":
            reject = True
            reasons.append("boolean_like")

        if cfg.reject_free_text and looks_free_text:
            reject = True
            reasons.append("free_text_like")

        if cfg.reject_measure_like and looks_measure_like and not key_name_hint:
            reject = True
            reasons.append("measure_like_name")

        if cfg.reject_date_like_name and looks_date_like_name and not key_name_hint:
            reject = True
            reasons.append("date_like_name")

        if cfg.reject_encoded_payload and looks_encoded_payload:
            reject = True
            reasons.append("encoded_payload_like")

        if looks_negative_name and not key_name_hint and dtype_family == "string" and avg_len >= 30.0:
            reject = True
            reasons.append("negative_name_hint")

        if not reject:
            if key_name_hint:
                reasons.append("key_name_hint")
            if small_domain:
                reasons.append("small_domain")
            if not reasons:
                reasons.append("keep")

        rows.append(
            {
                "table_name": table_name,
                "column_name": column_name,
                #"dtype_family": dtype_family,
                "reject": bool(reject),
                "key_name_hint": bool(key_name_hint),
                "small_domain": bool(small_domain),
                "looks_encoded_payload": bool(looks_encoded_payload),
                "looks_free_text": bool(looks_free_text),
                "looks_measure_like": bool(looks_measure_like),
                "name_tokens": tokens,
                "reasons": reasons,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(["table_name", "column_name"], kind="mergesort").reset_index(drop=True)


def apply_pruning_to_profiles(
    profiles_df: pd.DataFrame,
    pruned_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge pruning results back into profiles_df so downstream stages can
    filter once and reuse the same flags.
    """
    if pruned_df is None or pruned_df.empty:
        out = profiles_df.copy()
        out["reject"] = False
        out["key_name_hint"] = False
        out["small_domain"] = False
        out["looks_encoded_payload"] = False
        out["looks_free_text"] = False
        out["looks_measure_like"] = False
        out["name_tokens"] = [[] for _ in range(len(out))]
        out["reasons"] = [["keep"] for _ in range(len(out))]
        return out

    merge_cols = ["table_name", "column_name"]

    out = profiles_df.merge(
        pruned_df,
        on=merge_cols,
        how="left",
    )

    out["reject"] = out["reject"].fillna(False).astype(bool)
    out["key_name_hint"] = out["key_name_hint"].fillna(False).astype(bool)
    out["small_domain"] = out["small_domain"].fillna(False).astype(bool)
    out["looks_encoded_payload"] = out["looks_encoded_payload"].fillna(False).astype(bool)
    out["looks_free_text"] = out["looks_free_text"].fillna(False).astype(bool)
    out["looks_measure_like"] = out["looks_measure_like"].fillna(False).astype(bool)

    def _fill_list_col(series: pd.Series, default: list[str]) -> pd.Series:
        return series.apply(lambda x: x if isinstance(x, list) else default)

    out["name_tokens"] = _fill_list_col(out.get("name_tokens"), [])
    out["reasons"] = _fill_list_col(out.get("reasons"), ["keep"])

    return out