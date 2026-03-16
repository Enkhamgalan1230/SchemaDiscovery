# schema_discovery/candidates/ind_unary_soft.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd

from schema_discovery.quality.key_representations import build_key_representations, KeyRepConfig
from schema_discovery.scoring.fk_score import name_similarity


# =============================================================================
# Config
# =============================================================================
@dataclass(frozen=True)
class SoftIndConfig:
    # -------------------------------------------------------------------------
    # Accept thresholds (soft mode)
    # -------------------------------------------------------------------------
    min_distinct_coverage: float = 0.80
    min_row_coverage: float = 0.80

    # Alternative acceptance to reduce false negatives on optional / messy FKs
    alt_min_distinct_coverage: float = 0.95
    alt_min_row_coverage: float = 0.60
    min_matched_rows: int = 200

    # -------------------------------------------------------------------------
    # General gating
    # -------------------------------------------------------------------------
    # Allow unary UCC columns to also be treated as FK candidates
    allow_ucc_children: bool = True
    min_non_null: int = 20

    # Parent source control
    parent_source: str = "ucc_plus_profile"  # "ucc_only" | "profile_keylike" | "ucc_plus_profile"

    # Parent key-like gating when using profiles
    min_parent_distinct: int = 10

    # FK candidate gating (dynamic)
    min_child_distinct: int = 10
    # keep 0.00 if you want maximum recall, but expect more junk
    min_child_unique_ratio: float = 0.00
    max_child_null_ratio: float = 0.95

    # PK sanity gating (UCC implies uniqueness, but exports can be messy)
    # Only applies to UCC-derived parents, NOT profile-derived parents
    min_parent_unique_ratio: float = 0.60
    min_parent_non_null_ratio: float = 0.30

    # -------------------------------------------------------------------------
    # Candidate routing to avoid cartesian blow up
    # -------------------------------------------------------------------------
    enable_name_routing: bool = True
    min_name_sim_for_routing: float = 0.20  # was 0.30
    max_parents_per_fk: int = 20

    # Fallback routing when naming is weak (prevents "0 pairs tested")
    enable_fallback_routing: bool = True
    fallback_parents_per_fk: int = 20

    # -------------------------------------------------------------------------
    # Guard against small-domain false positives
    # -------------------------------------------------------------------------
    small_domain_fk_distinct_max: int = 30
    small_domain_min_name_sim: float = 0.65
    # Only apply the small-domain guard when FK column looks categorical (dominated)
    small_domain_top1_ratio_gate: float = 0.50

    # -------------------------------------------------------------------------
    # Key name hint (generic, dataset agnostic)
    # -------------------------------------------------------------------------
    key_name_hint_pattern: str = (
        r"(?:^id$|id$|_id$|key$|_key$|code$|_code$|ref$|_ref$|seq$|_seq$|no$|_no$|num$|_num$|uuid$|guid$)"
    )

    # -------------------------------------------------------------------------
    # Top1 dominance gate for columns (both children and profile-parents)
    # -------------------------------------------------------------------------
    top1_gate_distinct_max: int = 50
    max_top1_ratio: float = 0.90

    # -------------------------------------------------------------------------
    # Placeholder handling for coverage
    # -------------------------------------------------------------------------
    placeholder_tokens: tuple[str, ...] = (
        "0", "00", "000", "999", "9999", "999999",
        "unknown", "UNKNOWN",
        "null", "NULL",
    )
    placeholder_drop_top1_ratio: float = 0.75

    # Representation config
    rep_cfg: KeyRepConfig = field(default_factory=KeyRepConfig)


# =============================================================================
# Core coverage + range helpers
# =============================================================================
def _drop_dominant_placeholder(
    child: pd.Series,
    parent_set: set[str],
    *,
    max_top1_ratio: float,
    placeholder_drop_top1_ratio: float,
    placeholder_tokens: tuple[str, ...],
) -> pd.Series:
    """
    If FK is dominated by one value, treat it as a placeholder and drop it
    for coverage calculations.

    We drop when:
      1) top1_ratio >= placeholder_drop_top1_ratio AND top_val is a known placeholder token
      OR
      2) top1_ratio >= max_top1_ratio AND top_val is not in parent_set
    """
    c = child.dropna().astype("string")
    if c.empty:
        return c

    vc = c.value_counts(dropna=True)
    if vc.empty:
        return c

    top_val = str(vc.index[0])
    top_ratio = float(vc.iloc[0]) / float(len(c))

    norm_top = top_val.strip()
    norm_low = norm_top.lower()
    placeholder_low = {t.lower() for t in placeholder_tokens}

    is_placeholder = (norm_top in placeholder_tokens) or (norm_low in placeholder_low)

    if top_ratio >= float(placeholder_drop_top1_ratio) and is_placeholder:
        return c[c != top_val]

    if top_ratio >= float(max_top1_ratio) and top_val not in parent_set:
        return c[c != top_val]

    return c


def _coverage(
    child: pd.Series,
    parent_set: set[str],
    *,
    max_top1_ratio: float,
    placeholder_drop_top1_ratio: float,
    placeholder_tokens: tuple[str, ...],
) -> Tuple[float, float, int, int, int, int]:
    """
    Compute:
      distinct_coverage = |distinct(child) ∩ parent| / |distinct(child)|
      row_coverage      = matched_rows / non_null_rows

    Returns:
      distinct_cov, row_cov, matched_rows, fk_non_null_rows, fk_distinct, intersection_distinct
    """
    c = child.dropna().astype("string")
    if c.empty or not parent_set:
        return 0.0, 0.0, 0, int(len(c)), int(c.nunique(dropna=True)), 0

    c = _drop_dominant_placeholder(
        c,
        parent_set,
        max_top1_ratio=max_top1_ratio,
        placeholder_drop_top1_ratio=placeholder_drop_top1_ratio,
        placeholder_tokens=placeholder_tokens,
    )
    if c.empty:
        return 0.0, 0.0, 0, 0, 0, 0

    matched_mask = c.isin(parent_set)
    matched_rows = int(matched_mask.sum())

    fk_non_null_rows = int(len(c))
    row_cov = matched_rows / float(fk_non_null_rows) if fk_non_null_rows else 0.0

    fk_unique = pd.Index(c.unique())
    fk_distinct = int(len(fk_unique))
    if fk_distinct == 0:
        return 0.0, row_cov, matched_rows, fk_non_null_rows, 0, 0

    in_parent = fk_unique.isin(parent_set)
    intersection_distinct = int(in_parent.sum())
    distinct_cov = float(in_parent.mean())

    return distinct_cov, row_cov, matched_rows, fk_non_null_rows, fk_distinct, intersection_distinct


def _numeric_minmax(series: pd.Series) -> tuple[float | None, float | None]:
    """
    Best effort numeric min/max. Used only for range penalty in scoring.
    If cannot be safely parsed -> (None, None).
    """
    s = series.dropna()
    if s.empty:
        return None, None

    tok = s.astype("string").str.strip()
    num = pd.to_numeric(tok, errors="coerce").dropna()
    if num.empty:
        return None, None

    return float(num.min()), float(num.max())


def best_representation_match(
    fk_series: pd.Series,
    pk_series: pd.Series,
    cfg: SoftIndConfig,
) -> Dict[str, Any]:
    """
    Try all FK x PK representation pairs, choose best by:
      1) highest distinct_coverage
      2) tie break -> highest row_coverage
    """
    fk_reps = build_key_representations(fk_series, other=pk_series, cfg=cfg.rep_cfg)
    pk_reps = build_key_representations(pk_series, other=fk_series, cfg=cfg.rep_cfg)

    # Cache parent sets and distinct counts per PK rep
    pk_sets: Dict[str, set[str]] = {}
    pk_distincts: Dict[str, int] = {}
    for pk_name, pk_r in pk_reps.items():
        pv = pk_r.dropna().astype("string")
        s = set(pv.unique())
        pk_sets[pk_name] = s
        pk_distincts[pk_name] = int(len(s))

    best: Dict[str, Any] = {
        "fk_rep": None,
        "pk_rep": None,
        "distinct_coverage": 0.0,
        "row_coverage": 0.0,
        "matched_rows": 0,
        "fk_non_null_rows": 0,
        "fk_distinct": 0,
        "intersection_distinct": 0,
        "pk_distinct": 0,
    }

    for fk_name, fk_r in fk_reps.items():
        fk_nn = int(fk_r.notna().sum())
        if fk_nn == 0:
            continue

        for pk_name, parent_set in pk_sets.items():
            if not parent_set:
                continue

            dc, rc, mr, fk_nnz, fk_dist, inter = _coverage(
                fk_r,
                parent_set,
                max_top1_ratio=cfg.max_top1_ratio,
                placeholder_drop_top1_ratio=cfg.placeholder_drop_top1_ratio,
                placeholder_tokens=cfg.placeholder_tokens,
            )

            if (dc > best["distinct_coverage"]) or (dc == best["distinct_coverage"] and rc > best["row_coverage"]):
                best = {
                    "fk_rep": fk_name,
                    "pk_rep": pk_name,
                    "distinct_coverage": float(dc),
                    "row_coverage": float(rc),
                    "matched_rows": int(mr),
                    "fk_non_null_rows": int(fk_nnz),
                    "fk_distinct": int(fk_dist),
                    "intersection_distinct": int(inter),
                    "pk_distinct": int(pk_distincts.get(pk_name, 0)),
                }

    return best


# =============================================================================
# Candidate selection
# =============================================================================
def _build_parent_candidates(
    ucc_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    cfg: SoftIndConfig,
) -> pd.DataFrame:
    """
    Parents from unary UCCs + sanity gating using profiles.
    """
    if ucc_df is None or ucc_df.empty:
        return pd.DataFrame(columns=["table_name", "column_name"])

    if "is_unary_ucc" in ucc_df.columns:
        parents = ucc_df.loc[ucc_df["is_unary_ucc"] == True, ["table_name", "column_name"]].drop_duplicates()  # noqa: E712
    else:
        parents = ucc_df.loc[:, ["table_name", "column_name"]].drop_duplicates()

    if profiles_df is None or profiles_df.empty:
        return parents

    need = {"table_name", "column_name", "unique_ratio", "null_ratio", "n_rows"}
    if not need.issubset(set(profiles_df.columns)):
        return parents

    parents = parents.merge(
        profiles_df[list(need)],
        on=["table_name", "column_name"],
        how="left",
    )

    parents = parents.loc[
        (parents["n_rows"] >= cfg.min_non_null)
        & ((1.0 - parents["null_ratio"]) >= cfg.min_parent_non_null_ratio)
        & (parents["unique_ratio"] >= cfg.min_parent_unique_ratio),
        ["table_name", "column_name"],
    ].drop_duplicates()

    return parents


def _build_parent_candidates_from_profiles(profiles_df: pd.DataFrame, cfg: SoftIndConfig) -> pd.DataFrame:
    """
    Additional parents from profiling for messy exports where "true parent"
    columns might not be unique.
    """
    if profiles_df is None or profiles_df.empty:
        return pd.DataFrame(columns=["table_name", "column_name"])

    required = {"table_name", "column_name", "n_rows", "n_unique", "null_ratio", "top1_ratio"}
    missing = required - set(profiles_df.columns)
    if missing:
        raise ValueError(f"profiles_df missing required columns for soft IND parents: {sorted(missing)}")

    name_hint = profiles_df["column_name"].astype(str).str.contains(
        cfg.key_name_hint_pattern, case=False, regex=True
    )

    cand = profiles_df.loc[
        (profiles_df["n_rows"] >= cfg.min_non_null)
        & ((1.0 - profiles_df["null_ratio"]) >= cfg.min_parent_non_null_ratio)
        & (profiles_df["n_unique"] >= cfg.min_parent_distinct)
        & (name_hint),
        ["table_name", "column_name", "n_unique", "top1_ratio"],
    ].drop_duplicates()

    # Gate out categorical parents dominated by 1 value (only for small distinct)
    low_distinct = cand["n_unique"].astype(float) <= float(cfg.top1_gate_distinct_max)
    dominated = cand["top1_ratio"].astype(float) > float(cfg.max_top1_ratio)
    cand = cand[~(low_distinct & dominated)].copy()

    return cand[["table_name", "column_name"]].drop_duplicates()


def _build_child_candidates(
    profiles_df: pd.DataFrame,
    ucc_df: pd.DataFrame,
    cfg: SoftIndConfig,
) -> pd.DataFrame:
    """
    Children are FK-like columns from profiling.
    Excludes unary UCC columns to avoid reversed edges.
    """
    if profiles_df is None or profiles_df.empty:
        return pd.DataFrame(columns=["table_name", "column_name"])

    required = {"table_name", "column_name", "n_rows", "n_unique", "unique_ratio", "null_ratio", "top1_ratio"}
    missing = required - set(profiles_df.columns)
    if missing:
        raise ValueError(f"profiles_df missing required columns for soft IND: {sorted(missing)}")

    name_hint = profiles_df["column_name"].astype(str).str.contains(
        cfg.key_name_hint_pattern, case=False, regex=True
    )

    cand = profiles_df.loc[
        (profiles_df["n_rows"] >= cfg.min_non_null)
        & (profiles_df["null_ratio"] <= cfg.max_child_null_ratio)
        & (profiles_df["n_unique"] >= cfg.min_child_distinct)
        & (profiles_df["unique_ratio"] >= cfg.min_child_unique_ratio)
        & (name_hint),
        ["table_name", "column_name", "n_unique", "top1_ratio"],
    ].drop_duplicates()

    # Top1 dominance gate (small-domain + dominated)
    low_distinct = cand["n_unique"].astype(float) <= float(cfg.top1_gate_distinct_max)
    dominated = cand["top1_ratio"].astype(float) > float(cfg.max_top1_ratio)
    cand = cand[~(low_distinct & dominated)].copy()

    
    if (not cfg.allow_ucc_children) and (ucc_df is not None and not ucc_df.empty):
        ucc_pairs = set(zip(ucc_df["table_name"].astype(str), ucc_df["column_name"].astype(str)))
        cand_pairs = list(zip(cand["table_name"].astype(str), cand["column_name"].astype(str)))
        keep_mask = [p not in ucc_pairs for p in cand_pairs]
        cand = cand.loc[keep_mask].copy()

    return cand[["table_name", "column_name"]].drop_duplicates()


def _route_pairs(
    children: pd.DataFrame,
    parents: pd.DataFrame,
    cfg: SoftIndConfig,
) -> List[tuple[str, str, str, str, float]]:
    """
    Route FK candidates to a small set of PK candidates using name similarity.
    Returns list of (fk_table, fk_col, pk_table, pk_col, name_sim).
    """
    if children.empty or parents.empty:
        return []

    child_rows = [(r.table_name, r.column_name) for r in children.itertuples(index=False)]
    parent_rows = [(r.table_name, r.column_name) for r in parents.itertuples(index=False)]

    pairs: List[tuple[str, str, str, str, float]] = []

    if not cfg.enable_name_routing:
        for fk_t, fk_c in child_rows:
            for pk_t, pk_c in parent_rows:
                if fk_t != pk_t:
                    pairs.append((fk_t, fk_c, pk_t, pk_c, 0.0))
        return pairs

    for fk_t, fk_c in child_rows:
        scored: List[tuple[str, str, float]] = []
        for pk_t, pk_c in parent_rows:
            if fk_t == pk_t:
                continue
            sim = float(name_similarity(fk_c, pk_c))
            if sim >= float(cfg.min_name_sim_for_routing):
                scored.append((pk_t, pk_c, sim))

        if scored:
            scored.sort(key=lambda x: x[2], reverse=True)
            for pk_t, pk_c, sim in scored[: int(cfg.max_parents_per_fk)]:
                pairs.append((fk_t, fk_c, pk_t, pk_c, sim))
            continue

        # Fallback: if routing produced nothing, still test a limited set of parents
        if cfg.enable_fallback_routing:
            def _table_affinity(pk_t: str) -> int:
                return 1 if str(pk_t).lower().startswith(str(fk_t).lower()[:3]) else 0

            fallback = [(pk_t, pk_c) for pk_t, pk_c in parent_rows if pk_t != fk_t]
            fallback.sort(key=lambda x: _table_affinity(x[0]), reverse=True)

            for pk_t, pk_c in fallback[: int(cfg.fallback_parents_per_fk)]:
                pairs.append((fk_t, fk_c, pk_t, pk_c, 0.0))

    return pairs


# =============================================================================
# Public API
# =============================================================================
def discover_ind_unary_soft_all_pairs(
    *,
    dfs: Dict[str, pd.DataFrame],
    profiles_df: pd.DataFrame,
    ucc_df: pd.DataFrame,
    cfg: Optional[SoftIndConfig] = None,
    reject_log_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Soft IND discovery (routed pairs + representation matching).

    Output schema matches fk_score.required columns, so strict and soft
    can share the same scoring + selection.

    If reject_log_path is provided, writes a CSV of rejected routed pairs with reasons.
    """
    cfg = cfg or SoftIndConfig()

    if cfg.parent_source == "profile_keylike":
        parents = _build_parent_candidates_from_profiles(profiles_df, cfg)
    elif cfg.parent_source == "ucc_plus_profile":
        p1 = _build_parent_candidates(ucc_df, profiles_df, cfg)
        p2 = _build_parent_candidates_from_profiles(profiles_df, cfg)
        parents = pd.concat([p1, p2], ignore_index=True).drop_duplicates()
    else:
        parents = _build_parent_candidates(ucc_df, profiles_df, cfg)

    children = _build_child_candidates(profiles_df, ucc_df, cfg)
    cand_pairs = _route_pairs(children, parents, cfg)

    out_cols = [
        "fk_table", "fk_column", "pk_table", "pk_column",
        "distinct_coverage", "row_coverage",
        "fk_distinct", "pk_distinct", "intersection_distinct",
        "fk_min", "fk_max", "pk_min", "pk_max",
        "matched_rows", "fk_non_null_rows",
        "fk_rep", "pk_rep",
        "mode",
    ]

    if not cand_pairs:
        return pd.DataFrame(columns=out_cols)

    out_rows: list[dict[str, Any]] = []
    reject_rows: list[dict[str, Any]] = []

    def _reject(fk_t: str, fk_c: str, pk_t: str, pk_c: str, ns: float, reason: str) -> None:
        reject_rows.append(
            {
                "fk_table": fk_t,
                "fk_column": fk_c,
                "pk_table": pk_t,
                "pk_column": pk_c,
                "name_sim": float(ns),
                "reason": str(reason),
            }
        )

    for fk_table, fk_col, pk_table, pk_col, ns in cand_pairs:
        if fk_table not in dfs or pk_table not in dfs:
            _reject(fk_table, fk_col, pk_table, pk_col, ns, "table_missing")
            continue
        if fk_col not in dfs[fk_table].columns or pk_col not in dfs[pk_table].columns:
            _reject(fk_table, fk_col, pk_table, pk_col, ns, "column_missing")
            continue

        fk_s = dfs[fk_table][fk_col]
        pk_s = dfs[pk_table][pk_col]

        def _effective_non_null(s: pd.Series) -> int:
            x = s.astype("string").str.strip()
            x = x.where(x.str.len() > 0)
            return int(x.notna().sum())

        fk_eff = _effective_non_null(fk_s)
        pk_eff = _effective_non_null(pk_s)

        if fk_eff < int(cfg.min_non_null) or pk_eff < int(cfg.min_non_null):
            _reject(fk_table, fk_col, pk_table, pk_col, ns, "min_non_null_effective")
            continue

        best = best_representation_match(fk_s, pk_s, cfg)

        if best["fk_rep"] is None or best["pk_rep"] is None:
            _reject(
                fk_table, fk_col, pk_table, pk_col, ns,
                f"no_representation_match fk_eff={fk_eff} pk_eff={pk_eff}"
            )
            continue

        # Small-domain guard: only reject if column behaves categorical (dominated) AND naming is weak
        if int(best["fk_distinct"]) <= int(cfg.small_domain_fk_distinct_max):
            fk_reps_tmp = build_key_representations(fk_s, other=pk_s, cfg=cfg.rep_cfg)
            fk_tmp = fk_reps_tmp.get(best["fk_rep"], fk_s).dropna().astype("string")

            top1_ratio = 0.0
            if len(fk_tmp):
                vc = fk_tmp.value_counts(dropna=True)
                if not vc.empty:
                    top1_ratio = float(vc.iloc[0]) / float(len(fk_tmp))

            if top1_ratio >= float(cfg.small_domain_top1_ratio_gate) and float(ns) < float(cfg.small_domain_min_name_sim):
                _reject(fk_table, fk_col, pk_table, pk_col, ns, "small_domain_guard")
                continue

        # Coverage acceptance (primary + alternative rules)
        dc = float(best["distinct_coverage"])
        rc = float(best["row_coverage"])
        mr = int(best["matched_rows"])

        accept = False

        # Primary
        if dc >= float(cfg.min_distinct_coverage) and rc >= float(cfg.min_row_coverage):
            accept = True

        # Alt 1: very strong distinct evidence, allow lower row coverage
        if (not accept) and dc >= float(cfg.alt_min_distinct_coverage) and rc >= float(cfg.alt_min_row_coverage):
            accept = True

        # Alt 2: many matched rows, even if row coverage is depressed by optionality
        if (not accept) and dc >= float(cfg.min_distinct_coverage) and mr >= int(cfg.min_matched_rows):
            accept = True

        if not accept:
            _reject(fk_table, fk_col, pk_table, pk_col, ns, "coverage_fail")
            continue

        # Compute numeric ranges from winning representations (for range penalty)
        fk_reps = build_key_representations(fk_s, other=pk_s, cfg=cfg.rep_cfg)
        pk_reps = build_key_representations(pk_s, other=fk_s, cfg=cfg.rep_cfg)

        fk_best_s = fk_reps.get(best["fk_rep"], fk_s)
        pk_best_s = pk_reps.get(best["pk_rep"], pk_s)

        fk_min, fk_max = _numeric_minmax(fk_best_s)
        pk_min, pk_max = _numeric_minmax(pk_best_s)

        out_rows.append(
            {
                "fk_table": fk_table,
                "fk_column": fk_col,
                "pk_table": pk_table,
                "pk_column": pk_col,
                "distinct_coverage": float(best["distinct_coverage"]),
                "row_coverage": float(best["row_coverage"]),
                "fk_distinct": int(best["fk_distinct"]),
                "pk_distinct": int(best["pk_distinct"]),
                "intersection_distinct": int(best["intersection_distinct"]),
                "fk_min": fk_min,
                "fk_max": fk_max,
                "pk_min": pk_min,
                "pk_max": pk_max,
                "matched_rows": int(best["matched_rows"]),
                "fk_non_null_rows": int(best["fk_non_null_rows"]),
                "fk_rep": str(best["fk_rep"]),
                "pk_rep": str(best["pk_rep"]),
                "mode": "soft",
            }
        )

    # Optional reject log
    if reject_log_path is not None:
        try:
            pd.DataFrame(reject_rows).to_csv(reject_log_path, index=False)
        except Exception:
            pass

    if not out_rows:
        return pd.DataFrame(columns=out_cols)

    return pd.DataFrame(out_rows)[out_cols]