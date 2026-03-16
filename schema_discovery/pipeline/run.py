# schema_discovery/pipeline/run.py

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Any, Literal
from time import perf_counter

import pandas as pd

# Old pandas profiler path
from schema_discovery.profiling.profiler import profile_all_tables

# New DuckDB profiler path
from schema_discovery.profiling.profiler_sql import profile_all_tables_sql
from schema_discovery.storage import DuckDBTableStore

from schema_discovery.candidates.ucc_unary import discover_ucc_unary

# Old pandas strict + soft
from schema_discovery.candidates.ind_unary import discover_ind_unary
from schema_discovery.candidates.ind_unary_soft import (
    discover_ind_unary_soft_all_pairs as discover_ind_unary_soft_all_pairs_pandas,
    SoftIndConfig,
)

from schema_discovery.candidates.ind_unary_soft_sql import (
    discover_ind_unary_soft_all_pairs as discover_ind_unary_soft_all_pairs_sql,
)

from schema_discovery.pruning.pruning import PruningConfig, prune_profiles, apply_pruning_to_profiles

# New DuckDB strict path
from schema_discovery.candidates.ind_unary_sql import discover_ind_unary_sql

from schema_discovery.scoring.fk_score import score_ind_candidates
from schema_discovery.selection.select_edges import select_best_edges

from schema_discovery.quality.relational_missing import (
    RelMissingConfig,
    run_rel_missing_for_edge,
    rel_missing_summary_to_df,
)

from schema_discovery.quality.duplicates import (
    DuplicateChecksConfig,
    run_duplicate_checks,
)

from schema_discovery.quality.default_fk import (
    DefaultFKConfig,
    run_default_fk_for_edges,
    default_fk_summary_to_df,
)

from schema_discovery.annotation.surrogate_keys import (
    SurrogateKeyConfig,
    annotate_surrogate_keys,
)

from schema_discovery.candidates.ucc_composite import (
    CompositeUCCConfig,
    discover_ucc_composite,
)

from schema_discovery.recommendation import build_schema_recommendations


ProfileMode = Literal["basic", "enhanced"]
IndMode = Literal["strict", "soft"]


@dataclass
class PipelineConfig:
    # profiling
    sample_k: int = 5
    profile_mode: ProfileMode = "basic"  # basic | enhanced

    # UCC (unary)
    ucc_unique_ratio_min: float = 1.0
    ucc_null_ratio_max: float = 0.0

    # IND mode switch
    ind_mode: IndMode = "strict"

    # pruning step
    pruning_enabled: bool = True
    pruning_cfg: Optional[PruningConfig] = None

    # Shared IND gating
    ind_min_non_null_rows: int = 20

    # Strict IND inputs
    ind_min_distinct_child: int = 10
    ind_min_distinct_coverage: float = 0.90
    ind_dtype_families: tuple[str, ...] = ("int", "string")

    # Soft IND thresholds
    soft_ind_min_distinct_coverage: float = 0.80
    soft_ind_min_row_coverage: float = 0.80

    # Soft IND routing controls
    soft_ind_min_name_sim_for_routing: float = 0.30
    soft_ind_max_parents_per_fk: int = 20

    # Soft IND false positive controls
    soft_ind_small_domain_fk_distinct_max: int = 30
    soft_ind_small_domain_min_name_sim: float = 0.65

    # scoring
    w_distinct: float = 0.75
    w_row: float = 0.20
    w_name: float = 0.05
    w_range_penalty: float = 0.10
    fk_distinct_small_max: int = 2000

    # selection
    min_edge_score: float = 0.90

    # relational missingness
    rel_missing_enabled: bool = False
    rel_missing_sample_n: int = 10
    rel_orphan_warn_ratio: float = 0.01
    rel_orphan_fail_ratio: float = 0.05
    rel_missing_warn_ratio: float = 0.05
    rel_missing_fail_ratio: float = 0.20

    # duplicates
    duplicates_enabled: bool = False

    # default fk detection
    default_fk_enabled: bool = False
    default_fk_sample_n: int = 5
    default_fk_cfg: Optional[DefaultFKConfig] = None

    # duplicates config inputs
    duplicate_subset_keys: Optional[dict[str, list[list[str]]]] = None
    duplicate_relationship_tables: Optional[dict[str, dict[str, str]]] = None

    # surrogate key detection
    surrogate_keys_enabled: bool = False
    surrogate_keys_cfg: Optional[SurrogateKeyConfig] = None

    # composite ucc detection
    composite_ucc_enabled: bool = False
    composite_ucc_max_k: int = 3
    composite_ucc_max_cols_per_table: int = 12


def _validate_inputs(
    dfs: Optional[dict[str, pd.DataFrame]],
    store: Optional[DuckDBTableStore],
) -> None:
    if dfs is None and store is None:
        raise ValueError("Provide either dfs or store")

    if dfs is not None and not isinstance(dfs, dict):
        raise TypeError("dfs must be a dict[str, pd.DataFrame] when provided")

    if store is not None and not isinstance(store, DuckDBTableStore):
        raise TypeError("store must be a DuckDBTableStore when provided")


def _requires_in_memory_dfs(cfg: PipelineConfig, include: set[str]) -> list[str]:
    """
    These stages still depend on dfs-based modules.
    Keep this explicit so the transition is honest.
    """
    needs: list[str] = []

    if cfg.rel_missing_enabled:
        needs.append("relational_missing")

    if cfg.duplicates_enabled:
        needs.append("duplicates")

    if cfg.default_fk_enabled:
        needs.append("default_fk")

    if cfg.surrogate_keys_enabled:
        needs.append("surrogate_keys")

    if cfg.composite_ucc_enabled:
        needs.append("composite_ucc")

    if "recommendations" in include:
        needs.append("recommendations")

    return needs


def run_schema_discovery(
    dfs: Optional[dict[str, pd.DataFrame]] = None,
    *,
    store: Optional[DuckDBTableStore] = None,
    config: Optional[PipelineConfig] = None,
    include_stages: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Transitional orchestration layer.

    Supported modes:
    1. dfs only
       -> old full pandas path works as before

    2. store only
       -> SQL profiler + SQL strict IND or hybrid soft IND + scoring + selection
       -> downstream stages that still require dfs are blocked clearly

    3. dfs + store
       -> profiling / strict IND can use store
       -> downstream dfs stages still allowed
       -> useful during migration / A-B testing
    """
    _validate_inputs(dfs=dfs, store=store)

    cfg_in = config or PipelineConfig()
    include = set(include_stages or [])

    if "recommendations" in include:
        cfg = replace(
            cfg_in,
            rel_missing_enabled=True,
            duplicates_enabled=True,
            default_fk_enabled=True,
            surrogate_keys_enabled=True,
            composite_ucc_enabled=True,
        )
    else:
        cfg = cfg_in

    using_store = store is not None
    using_dfs = dfs is not None

    runtime_summary: dict[str, float] = {}
    metric_summary: dict[str, Any] = {}
    pipeline_start = perf_counter()

    # ------------------------------------------------------------
    # Guard unsupported store-only stages
    # ------------------------------------------------------------
    if using_store and not using_dfs:
        needs_dfs = _requires_in_memory_dfs(cfg, include)
        if needs_dfs:
            raise NotImplementedError(
                "Store-only mode is currently implemented for profiling, unary UCC, "
                "strict unary IND, soft IND, scoring, and edge selection. "
                f"These requested stages still require dfs in memory: {sorted(needs_dfs)}"
            )

    # ------------------------------------------------------------
    # 1) profiling
    # ------------------------------------------------------------
    t0 = perf_counter()
    if using_store:
        profiles_df, profile_extras = profile_all_tables_sql(
            store,
            sample_k=cfg.sample_k,
        )
    else:
        mode: ProfileMode = cfg.profile_mode if cfg.profile_mode in ("basic", "enhanced") else "basic"
        profiles_df, profile_extras = profile_all_tables(
            dfs,
            sample_k=cfg.sample_k,
            mode=mode,
        )
    runtime_summary["profiling_seconds"] = perf_counter() - t0
    metric_summary["profiled_columns"] = int(len(profiles_df)) if profiles_df is not None else 0
    metric_summary["profiled_tables"] = (
        int(profiles_df["table_name"].nunique())
        if profiles_df is not None and not profiles_df.empty
        else 0
    )

    # ------------------------------------------------------------
    # 2) UCC (unary)
    # ------------------------------------------------------------
    t0 = perf_counter()
    ucc_df = discover_ucc_unary(
        profiles_df,
        unique_ratio_min=cfg.ucc_unique_ratio_min,
        null_ratio_max=cfg.ucc_null_ratio_max,
    )
    runtime_summary["ucc_seconds"] = perf_counter() - t0
    metric_summary["ucc_columns"] = int(len(ucc_df)) if ucc_df is not None else 0

    # ------------------------------------------------------------
    # 2.5) pruning
    # ------------------------------------------------------------
    pruned_df = pd.DataFrame()
    t0 = perf_counter()
    if cfg.pruning_enabled:
        p_cfg = cfg.pruning_cfg or PruningConfig()
        pruned_df = prune_profiles(profiles_df, cfg=p_cfg)
        profiles_df = apply_pruning_to_profiles(profiles_df, pruned_df)
    runtime_summary["pruning_seconds"] = perf_counter() - t0

    if pruned_df is not None and not pruned_df.empty and "reject" in pruned_df.columns:
        rejected_cols = int(pruned_df["reject"].sum())
        kept_cols = int((~pruned_df["reject"]).sum())
    else:
        rejected_cols = 0
        kept_cols = int(len(profiles_df)) if profiles_df is not None else 0

    metric_summary["pruned_rejected_columns"] = rejected_cols
    metric_summary["pruned_kept_columns"] = kept_cols

    # ------------------------------------------------------------
    # 3) IND (strict OR soft, never both)
    # ------------------------------------------------------------
    t0 = perf_counter()
    if cfg.ind_mode == "soft":
        soft_cfg = SoftIndConfig(
            parent_source="ucc_plus_profile",
            min_distinct_coverage=cfg.soft_ind_min_distinct_coverage,
            min_row_coverage=cfg.soft_ind_min_row_coverage,
            min_non_null=cfg.ind_min_non_null_rows,
            min_name_sim_for_routing=cfg.soft_ind_min_name_sim_for_routing,
            max_parents_per_fk=cfg.soft_ind_max_parents_per_fk,
            small_domain_fk_distinct_max=cfg.soft_ind_small_domain_fk_distinct_max,
            small_domain_min_name_sim=cfg.soft_ind_small_domain_min_name_sim,
            allow_ucc_children=True,
        )

        if using_store and not using_dfs:
            ind_df = discover_ind_unary_soft_all_pairs_sql(
                profiles_df=profiles_df,
                ucc_df=ucc_df,
                cfg=soft_cfg,
                store=store,
            )
        else:
            ind_df = discover_ind_unary_soft_all_pairs_pandas(
                dfs=dfs,
                profiles_df=profiles_df,
                ucc_df=ucc_df,
                cfg=soft_cfg,
            )
    else:
        if using_store:
            ind_df = discover_ind_unary_sql(
                store=store,
                ucc_df=ucc_df,
                profiles_df=profiles_df,
                min_distinct_child=cfg.ind_min_distinct_child,
                min_non_null_rows=cfg.ind_min_non_null_rows,
                min_distinct_coverage=cfg.ind_min_distinct_coverage,
                dtype_families=cfg.ind_dtype_families,
            )
        else:
            ind_df = discover_ind_unary(
                dfs=dfs,
                ucc_df=ucc_df,
                profiles_df=profiles_df,
                min_distinct_child=cfg.ind_min_distinct_child,
                min_non_null_rows=cfg.ind_min_non_null_rows,
                min_distinct_coverage=cfg.ind_min_distinct_coverage,
                dtype_families=cfg.ind_dtype_families,
            )
    runtime_summary["ind_seconds"] = perf_counter() - t0
    metric_summary["ind_candidates"] = int(len(ind_df)) if ind_df is not None else 0

    # ------------------------------------------------------------
    # 4) scoring
    # ------------------------------------------------------------
    t0 = perf_counter()
    scored_df = (
        score_ind_candidates(
            ind_df,
            w_distinct=cfg.w_distinct,
            w_row=cfg.w_row,
            w_name=cfg.w_name,
            w_range_penalty=cfg.w_range_penalty,
            fk_distinct_small_max=cfg.fk_distinct_small_max,
        )
        if (ind_df is not None and not ind_df.empty)
        else (ind_df.copy() if ind_df is not None else pd.DataFrame())
    )
    runtime_summary["scoring_seconds"] = perf_counter() - t0
    metric_summary["scored_candidates"] = int(len(scored_df)) if scored_df is not None else 0

    # ------------------------------------------------------------
    # 5) selection
    # ------------------------------------------------------------
    t0 = perf_counter()
    edges_df = (
        select_best_edges(
            scored_df,
            profiles_df=profiles_df,
            min_score=cfg.min_edge_score,
        )
        if (scored_df is not None and not scored_df.empty)
        else (scored_df.copy() if scored_df is not None else pd.DataFrame())
    )
    runtime_summary["selection_seconds"] = perf_counter() - t0
    metric_summary["final_edges"] = int(len(edges_df)) if edges_df is not None else 0

    # ------------------------------------------------------------
    # 6) relational missingness (optional, dfs only for now)
    # ------------------------------------------------------------
    rel_missing_df = pd.DataFrame()
    if cfg.rel_missing_enabled and (edges_df is not None and not edges_df.empty):
        if not using_dfs:
            raise NotImplementedError("relational_missing still requires dfs in memory")

        cfg_rm = RelMissingConfig(
            orphan_warn_ratio=cfg.rel_orphan_warn_ratio,
            orphan_fail_ratio=cfg.rel_orphan_fail_ratio,
            missing_warn_ratio=cfg.rel_missing_warn_ratio,
            missing_fail_ratio=cfg.rel_missing_fail_ratio,
        )

        rel_items: list[dict[str, Any]] = []
        for edge in edges_df.to_dict(orient="records"):
            rel_items.append(
                run_rel_missing_for_edge(
                    dfs=dfs,
                    edge_row=edge,
                    profiles_df=profiles_df,
                    cfg=cfg_rm,
                    sample_n=cfg.rel_missing_sample_n,
                )
            )
        rel_missing_df = rel_missing_summary_to_df(rel_items)

    # ------------------------------------------------------------
    # 7) duplicates (optional, dfs only for now)
    # ------------------------------------------------------------
    dup_out: dict[str, pd.DataFrame] = {}
    if cfg.duplicates_enabled:
        if not using_dfs:
            raise NotImplementedError("duplicates checks still require dfs in memory")

        dup_cfg = DuplicateChecksConfig(
            relational_duplicate_subset_keys=(cfg.duplicate_subset_keys or {}),
            relationship_tables=(cfg.duplicate_relationship_tables or {}),
        )
        dup_out = run_duplicate_checks(dfs, cfg=dup_cfg)

    # ------------------------------------------------------------
    # 8) default fk values (optional, dfs only for now)
    # ------------------------------------------------------------
    default_fk_df = pd.DataFrame()
    default_fk_raw: list[dict[str, Any]] = []
    if cfg.default_fk_enabled and (edges_df is not None and not edges_df.empty):
        if not using_dfs:
            raise NotImplementedError("default_fk detection still requires dfs in memory")

        dfk_cfg = cfg.default_fk_cfg or DefaultFKConfig()
        default_fk_raw = run_default_fk_for_edges(
            dfs=dfs,
            edges_df=edges_df,
            cfg=dfk_cfg,
            sample_n=cfg.default_fk_sample_n,
        )
        default_fk_df = default_fk_summary_to_df(default_fk_raw)

    # ------------------------------------------------------------
    # 9) surrogate keys (optional, dfs only for now)
    # ------------------------------------------------------------
    surrogate_keys_out: dict[str, Any] = {"items": []}
    if cfg.surrogate_keys_enabled:
        if not using_dfs:
            raise NotImplementedError("surrogate key detection still requires dfs in memory")

        sk_cfg = cfg.surrogate_keys_cfg or SurrogateKeyConfig()
        surrogate_keys_out = annotate_surrogate_keys(
            dfs=dfs,
            profiles_df=profiles_df,
            edges_df=edges_df,
            ucc_df=ucc_df,
            cfg=sk_cfg,
        )

    # ------------------------------------------------------------
    # 10) composite UCC (optional, dfs only for now)
    # ------------------------------------------------------------
    composite_ucc_df = pd.DataFrame()
    if cfg.composite_ucc_enabled:
        if not using_dfs:
            raise NotImplementedError("composite UCC discovery still requires dfs in memory")

        composite_ucc_df = discover_ucc_composite(
            dfs=dfs,
            profiles_df=profiles_df,
            ucc_unary_df=ucc_df,
            edges_df=edges_df,
            cfg=CompositeUCCConfig(
                max_k=cfg.composite_ucc_max_k,
                max_cols_per_table=cfg.composite_ucc_max_cols_per_table,
            ),
        )

    # ------------------------------------------------------------
    # 11) recommendations (dfs only for now)
    # ------------------------------------------------------------
    out_bundle: dict[str, Any] = {
        "profiles_df": profiles_df,
        "profile_extras": profile_extras,
        "ucc_df": ucc_df,
        "composite_ucc_df": composite_ucc_df,
        "edges_df": edges_df,
        "rel_missing_df": rel_missing_df,
        "duplicates": dup_out,
        "default_fk_df": default_fk_df,
        "surrogate_keys": surrogate_keys_out,
    }

    if using_dfs:
        recommendations = build_schema_recommendations(
            dfs=dfs,
            out=out_bundle,
            normalisation_policy={
                "null_tokens_mapped": ["", "null", "none", "n/a", "na"],
                "trim_whitespace": True,
                "casefold_strings": True,
                "remove_punctuation": False,
            },
            top_n_alternatives=3,
        )
    else:
        recommendations = {}

    runtime_summary["total_seconds"] = perf_counter() - pipeline_start

    return {
        "profiles_df": profiles_df,
        "profile_extras": profile_extras,
        "ucc_df": ucc_df,
        "pruned_df": pruned_df,
        "ind_df": ind_df,
        "scored_df": scored_df,
        "edges_df": edges_df,
        "rel_missing_df": rel_missing_df,
        "duplicates": dup_out,
        "default_fk_df": default_fk_df,
        "default_fk_raw": default_fk_raw,
        "surrogate_keys": surrogate_keys_out,
        "composite_ucc_df": composite_ucc_df,
        "recommendations": recommendations,
        "backend": "duckdb" if using_store else "pandas",
        "runtime_summary": runtime_summary,
        "metric_summary": metric_summary,
    }