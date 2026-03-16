from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import re
import pandas as pd


_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}$"
)


@dataclass(frozen=True)
class SurrogateKeyConfig:
    # Strict key gate (PK like)
    unique_ratio_min: float = 0.999
    null_ratio_max: float = 0.01

    # Business key tier (near unique, not necessarily surrogate)
    business_unique_ratio_min: float = 0.98
    business_null_ratio_max: float = 0.05
    business_requires_name_hint: bool = True

    # Soft signals
    continuity_good: float = 0.80  # only used for int-like keys
    referenced_by_edges_good: int = 2

    # Penalise likely external/natural identifiers (barcode/sku/isbn/etc.)
    natural_id_penalty: float = 0.20  # multiplied by natural_id_hint in [0,1]

    # Name hint weight is intentionally low
    name_hint: bool = True

    # Relationship evidence filter (prevents false IND edges poisoning ref_counts)
    min_edge_score_for_ref: float = 0.95
    min_name_sim_for_ref: float = 0.30


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _is_uuid_like(series: pd.Series, sample_n: int = 200) -> bool:
    s = series.dropna()
    if s.empty:
        return False
    s = s.astype(str).head(sample_n)
    matches = s.map(lambda v: bool(_UUID_RE.match(v.strip())))
    return bool(matches.mean() >= 0.95)


def _continuity_int(series: pd.Series) -> float | None:
    """
    continuity = n_unique / (max - min + 1)
    Only meaningful for integer-like sequences.
    Allows gaps (continuity < 1.0).
    """
    s = series.dropna()
    if s.empty:
        return None

    try:
        x = pd.to_numeric(s, errors="coerce").dropna()
        if x.empty:
            return None

        # require integer-ish values
        if not (x % 1 == 0).all():
            return None

        x = x.astype("int64")
        n_unique = int(x.nunique())
        mn = int(x.min())
        mx = int(x.max())
        denom = (mx - mn + 1)
        if denom <= 0:
            return None
        return n_unique / denom
    except Exception:
        return None


def _name_hint(col: str) -> float:
    c = (col or "").strip().lower()
    if c == "id":
        return 1.0
    if c.endswith("_id"):
        return 0.7
    if "uuid" in c or "guid" in c:
        return 0.7
    return 0.0


def _business_key_name_hint(col: str) -> float:
    """
    Near-unique business identifiers you may want to surface even if not surrogate.
    Keep conservative.
    """
    c = (col or "").strip().lower()
    tokens = (
        "invoice", "inv",
        "order_number", "ordernumber",
        "number", "code",
        "reference", "ref",
        "promo", "email",
    )
    return 1.0 if any(t in c for t in tokens) else 0.0


def _is_descriptive_name(col: str) -> bool:
    c = (col or "").strip().lower()
    tokens = ("name", "description", "title", "text", "notes", "comment", "message")
    return any(t in c for t in tokens)


def _natural_id_hint(col: str) -> float:
    """
    Detect column names that are usually natural/external identifiers rather than
    database-generated surrogates. Returns strength in [0, 1].

    High precision is the goal -> keep this list conservative.
    """
    c = (col or "").strip().lower()

    strong_tokens = (
        "barcode", "ean", "upc", "gtin", "isbn", "asin", "vin",
        "passport", "ssn", "nin", "tax", "vat", "utr",
    )
    medium_tokens = (
        "sku", "product_code", "item_code",
        "part_no", "part_number",
        "serial", "serial_no", "serial_number",
        "reference", "ref",
    )

    if any(tok in c for tok in strong_tokens):
        return 1.0
    if any(tok in c for tok in medium_tokens):
        return 0.6
    return 0.0


def _edge_is_reliable_for_ref(r: dict[str, Any], cfg: SurrogateKeyConfig) -> bool:
    """
    Only count references from edges that are strong enough to trust.
    This prevents false IND matches inflating referenced_by_edges.
    """
    # score filter if present
    if "score" in r and r.get("score") is not None:
        try:
            if float(r["score"]) < float(cfg.min_edge_score_for_ref):
                return False
        except Exception:
            return False

    fk_col = str(r.get("fk_column", "")).lower()
    pk_col = str(r.get("pk_column", "")).lower()

    # name similarity filter if present
    if "name_sim" in r and r.get("name_sim") is not None:
        try:
            ns = float(r["name_sim"])
        except Exception:
            ns = 0.0

        id_like_pair = (fk_col.endswith("id") or fk_col.endswith("_id")) and (pk_col.endswith("id") or pk_col.endswith("_id"))
        if (ns < float(cfg.min_name_sim_for_ref)) and (not id_like_pair):
            return False

    # reject obvious measure-like columns unless they are explicit *_id
    bad_fk_tokens = ("minute", "second", "hour", "status", "flag", "qty", "count", "number_of")
    if any(t in fk_col for t in bad_fk_tokens) and not (fk_col.endswith("id") or fk_col.endswith("_id")):
        return False

    return True


def annotate_surrogate_keys(
    dfs: dict[str, pd.DataFrame],
    profiles_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    ucc_df: pd.DataFrame | None = None,
    cfg: SurrogateKeyConfig | None = None,
) -> dict[str, Any]:
    """
    Return surrogate key annotations.

    This does NOT mutate inputs and does NOT change edge selection.
    It only annotates likely keys and classifies them:
      - key_kind: surrogate | business_key | key_candidate

    Output schema:
    {
      "items": [
        {
          "table": str,
          "column": str,
          "key_role": "pk_candidate" | "key_candidate",
          "key_kind": "surrogate" | "business_key" | "key_candidate",
          "is_surrogate": bool,
          "confidence": float,
          "signals": {...}
        },
        ...
      ]
    }
    """
    if cfg is None:
        cfg = SurrogateKeyConfig()

    if profiles_df is None or profiles_df.empty:
        return {"items": []}

    # Build: referenced_by_edges count for each (pk_table, pk_column), but only from reliable edges
    ref_counts: dict[tuple[str, str], int] = {}
    if edges_df is not None and (not edges_df.empty):
        for r in edges_df.to_dict(orient="records"):
            if not _edge_is_reliable_for_ref(r, cfg):
                continue

            pk_t = r.get("pk_table")
            pk_c = r.get("pk_column")
            if pk_t is None or pk_c is None:
                continue
            key = (str(pk_t), str(pk_c))
            ref_counts[key] = ref_counts.get(key, 0) + 1

    # Optional: detect existence of alternative natural key candidates (same table)
    alt_ucc_by_table: dict[str, list[str]] = {}
    if ucc_df is not None and (not ucc_df.empty):
        for r in ucc_df.to_dict(orient="records"):
            t = str(r.get("table_name", ""))
            c = str(r.get("column_name", ""))
            if t and c:
                alt_ucc_by_table.setdefault(t, []).append(c)

    items: list[dict[str, Any]] = []

    pdf = profiles_df.copy()
    required_cols = {"table_name", "column_name", "unique_ratio", "null_ratio"}
    if not required_cols.issubset(set(pdf.columns)):
        return {"items": []}

    for row in pdf.to_dict(orient="records"):
        table = str(row.get("table_name"))
        col = str(row.get("column_name"))

        unique_ratio = _safe_float(row.get("unique_ratio"))
        null_ratio = _safe_float(row.get("null_ratio"))
        dtype_family = str(row.get("dtype_family", "")).lower()

        if unique_ratio is None or null_ratio is None:
            continue

        # Two-tier gate: strict key-like OR near-unique business-key-like
        is_strict_key_like = (unique_ratio >= float(cfg.unique_ratio_min)) and (null_ratio <= float(cfg.null_ratio_max))
        is_business_key_like = (unique_ratio >= float(cfg.business_unique_ratio_min)) and (null_ratio <= float(cfg.business_null_ratio_max))

        if not (is_strict_key_like or is_business_key_like):
            continue

        # Pull series if we need deeper signals
        series = dfs.get(table, pd.DataFrame()).get(col) if table in dfs and col in dfs[table].columns else None

        uuid_like = False
        continuity = None

        if series is not None:
            if dtype_family in ("string", "object"):
                uuid_like = _is_uuid_like(series)
            if dtype_family in ("int", "integer", "numeric"):
                continuity = _continuity_int(series)

        referenced_by = ref_counts.get((table, col), 0)
        name_score = _name_hint(col) if cfg.name_hint else 0.0
        natural_id_hint = _natural_id_hint(col)
        biz_hint = _business_key_name_hint(col)
        is_desc = _is_descriptive_name(col)

        alt_ucc_cols = alt_ucc_by_table.get(table, [])
        alt_natural_exists = any(
            (c.lower() not in ("id", col.lower())) and (not c.lower().endswith("_id"))
            for c in alt_ucc_cols
        )

        # Confidence scoring: base means "key-like"
        score = 0.50
        if uuid_like:
            score += 0.25
        if continuity is not None and continuity >= float(cfg.continuity_good):
            score += 0.15
        if referenced_by >= int(cfg.referenced_by_edges_good):
            score += 0.15
        if alt_natural_exists:
            score += 0.10
        if name_score > 0:
            score += min(0.10, 0.10 * float(name_score))

        # Penalise likely external/natural identifiers (barcode/sku/isbn/etc.)
        if natural_id_hint > 0:
            score -= float(cfg.natural_id_penalty) * float(natural_id_hint)

        # Penalise descriptive columns that are unique by construction
        if is_desc and referenced_by == 0 and name_score == 0.0 and not uuid_like:
            score -= 0.20

        # clamp
        score = float(max(0.0, min(1.0, score)))

        # Classification
        # Surrogate should be strict-key-like and should not be descriptive text
        is_surrogate = bool((score >= 0.75) and is_strict_key_like and (not is_desc))

        # Business key should be near-unique but not strict, and should have a name hint
        is_business_key = False
        if is_business_key_like and (not is_strict_key_like):
            if (not bool(cfg.business_requires_name_hint)) or (biz_hint >= 1.0):
                is_business_key = True

        key_kind = "surrogate" if is_surrogate else ("business_key" if is_business_key else "key_candidate")

        # Role based on reliable edges
        key_role = "pk_candidate" if referenced_by > 0 else "key_candidate"

        items.append(
            {
                "table": table,
                "column": col,
                "key_role": key_role,
                "key_kind": key_kind,
                "is_surrogate": is_surrogate,
                "confidence": round(float(score), 3),
                "signals": {
                    "dtype_family": dtype_family,
                    "unique_ratio": float(unique_ratio),
                    "null_ratio": float(null_ratio),
                    "uuid_like": bool(uuid_like),
                    "continuity": None if continuity is None else round(float(continuity), 3),
                    "referenced_by_edges": int(referenced_by),
                    "alt_natural_key_ucc_exists": bool(alt_natural_exists),
                    "name_hint": round(float(name_score), 3),
                    "business_key_name_hint": round(float(biz_hint), 3),
                    "is_descriptive_name": bool(is_desc),
                    "natural_id_hint": round(float(natural_id_hint), 3),
                    "natural_id_penalty": float(cfg.natural_id_penalty),
                    "ref_filter_min_edge_score": float(cfg.min_edge_score_for_ref),
                    "ref_filter_min_name_sim": float(cfg.min_name_sim_for_ref),
                },
            }
        )

    # Sort: pk candidates first, then confidence
    items.sort(
        key=lambda d: (
            0 if d["key_role"] == "pk_candidate" else 1,
            0 if d.get("key_kind") == "surrogate" else (1 if d.get("key_kind") == "business_key" else 2),
            -float(d["confidence"]),
            d["table"],
            d["column"],
        )
    )

    return {"items": items}
