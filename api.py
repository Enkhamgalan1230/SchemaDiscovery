from __future__ import annotations

# Schema Discovery API

import io
import time
from typing import Any, Optional, Literal, get_args

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from schema_discovery.pipeline.run import run_schema_discovery, PipelineConfig
from schema_discovery.viz.erd_graphviz import edges_to_dot

# Run:
#   .venv\Scripts\Activate
#   uvicorn api:app --reload --port 8000


# =============================================================================
# API contract
# =============================================================================
Stage = Literal[
    "profiles",
    "profiles_enhanced",
    "ucc",
    "ucc_composite",
    "ind",
    "scored",
    "relationships",
    "rel_missing",
    "duplicates",
    "duplicates_summary",
    "default_fk",
    "default_fk_summary",
    "surrogate_keys",
    "surrogate_keys_summary",
    "recommendations",
]
IndMode = Literal["strict", "soft"]

app = FastAPI(title="Schema Discovery API", version="1.0.0")

DEFAULT_INCLUDE: tuple[str, ...] = ("relationships",)
MAX_FILES_DEFAULT = 30

# Base columns only for "profiles" stage to keep payload small
BASE_PROFILE_COLS: tuple[str, ...] = (
    "table_name",
    "column_name",
    "column_key",
    "dtype",
    "dtype_family",
    "n_rows",
    "n_null",
    "n_non_null",
    "null_ratio",
    "n_unique",
    "unique_ratio",
    "unique_ratio_non_null",
    "is_unary_ucc",
    "sample_values",
    "avg_len",
    "min_len",
    "max_len",
)

# Defaults used for strict/soft param validation (keep in sync with Query defaults)
STRICT_DEFAULTS = {
    "min_distinct_coverage": 0.90,
    "min_row_coverage": 0.90,
}
SOFT_DEFAULTS = {
    "soft_min_row_coverage": 0.80,
    "soft_min_distinct_coverage": 0.80,
    "soft_min_name_sim_for_routing": 0.30,
    "soft_max_parents_per_fk": 20,
    "soft_small_domain_fk_distinct_max": 30,
    "soft_small_domain_min_name_sim": 0.65,
}


# =============================================================================
# Error payload helpers
# =============================================================================
def error_response(
    *,
    http_status: int,
    code: str,
    message: str,
    details: Optional[dict[str, Any]] = None,
) -> JSONResponse:
    payload: dict[str, Any] = {"error": {"code": code, "message": message}}
    if details:
        payload["error"]["details"] = details
    return JSONResponse(status_code=http_status, content=payload)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    # Keep a stable error payload shape.
    if isinstance(exc.detail, dict):
        return error_response(
            http_status=exc.status_code,
            code=exc.detail.get("code", "HTTP_ERROR"),
            message=exc.detail.get("message", "Request failed"),
            details=exc.detail.get("details"),
        )
    return error_response(http_status=exc.status_code, code="HTTP_ERROR", message=str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    return error_response(
        http_status=422,
        code="VALIDATION_ERROR",
        message="Request parameters failed validation",
        details={"errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    return error_response(
        http_status=500,
        code="INTERNAL_ERROR",
        message="Unexpected server error",
        details={"exception_type": type(exc).__name__, "exception_message": str(exc)},
    )


# =============================================================================
# Upload reading
# =============================================================================
def _safe_table_name(filename: str | None) -> str:
    # "orders.csv" -> "orders"
    name = (filename or "table").rsplit(".", 1)[0].strip()
    return name or "table"


def _read_csv_upload(f: UploadFile) -> pd.DataFrame:
    raw = f.file.read()
    if not raw:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "EMPTY_FILE",
                "message": "Uploaded file is empty",
                "details": {"filename": f.filename},
            },
        )

    try:
        return pd.read_csv(
            io.BytesIO(raw),
            dtype="string",
            keep_default_na=False,
            na_values=["", "NULL", "null", "None", "none", "N/A", "n/a", "NA", "na"],
            low_memory=False,
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "CSV_READ_FAILED",
                "message": "Failed to read CSV upload",
                "details": {"filename": f.filename, "reason": str(e)},
            },
        )


def _read_uploads(files: list[UploadFile], max_files: int = MAX_FILES_DEFAULT) -> dict[str, pd.DataFrame]:
    if not files:
        raise HTTPException(status_code=400, detail={"code": "NO_FILES", "message": "No CSV files provided"})

    if len(files) > max_files:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "TOO_MANY_FILES",
                "message": "Too many files provided",
                "details": {"max_files": max_files, "received": len(files)},
            },
        )

    dfs: dict[str, pd.DataFrame] = {}
    for f in files:
        dfs[_safe_table_name(f.filename)] = _read_csv_upload(f)
    return dfs


# =============================================================================
# JSON conversion helpers
# =============================================================================
def _json_safe(obj: Any) -> Any:
    if obj is None or obj is pd.NA:
        return None

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()

    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    return obj


def _df_to_records(df: pd.DataFrame, max_rows: int) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    df = df.head(int(max_rows)).copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    records = df.to_dict(orient="records")
    return [{str(k): _json_safe(v) for k, v in row.items()} for row in records]


# =============================================================================
# Request parsing
# =============================================================================
def _parse_include(include_csv: Optional[str], include_list: Optional[list[Stage]], debug_bundle: bool) -> list[str]:
    allowed = set(get_args(Stage))

    if include_list:
        # include_stage=relationships&include_stage=profiles
        return list(dict.fromkeys([str(s) for s in include_list]))

    if debug_bundle and not include_csv:
        return [
            "profiles",
            "profiles_enhanced",
            "ucc",
            "ucc_composite",
            "ind",
            "scored",
            "relationships",
            "rel_missing",
            "duplicates",
            "duplicates_summary",
            "default_fk",
            "default_fk_summary",
            "surrogate_keys",
            "surrogate_keys_summary",
            "recommendations",
        ]

    if not include_csv:
        return list(DEFAULT_INCLUDE)

    parts = [p.strip().lower() for p in include_csv.split(",") if p.strip()]
    unknown = [p for p in parts if p not in allowed]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "UNKNOWN_STAGE",
                "message": "Unknown include stage(s)",
                "details": {"unknown": unknown, "allowed": sorted(allowed)},
            },
        )
    return list(dict.fromkeys(parts))


# =============================================================================
# Profiles lookup helper
# =============================================================================
def _profile_lookup(profiles_df: pd.DataFrame, table: str, col: str) -> dict[str, Any]:
    if profiles_df is None or profiles_df.empty:
        return {}
    sub = profiles_df[(profiles_df["table_name"] == table) & (profiles_df["column_name"] == col)]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


# =============================================================================
# Relationships formatting
# =============================================================================
def _summarise_relationships(edges_df: pd.DataFrame) -> dict[str, Any]:
    if edges_df is None or edges_df.empty:
        return {"count": 0, "many_to_one": 0, "one_to_one": 0, "optional": 0, "required": 0}

    count = int(len(edges_df))
    many_to_one = int((edges_df.get("cardinality") == "many_to_one").sum()) if "cardinality" in edges_df.columns else 0
    one_to_one = int((edges_df.get("cardinality") == "one_to_one").sum()) if "cardinality" in edges_df.columns else 0
    optional = int(edges_df.get("optional", False).sum()) if "optional" in edges_df.columns else 0
    required = int(count - optional)

    return {
        "count": count,
        "many_to_one": many_to_one,
        "one_to_one": one_to_one,
        "optional": optional,
        "required": required,
    }


def _edges_to_relationship_items(
    edges_df: pd.DataFrame,
    *,
    profiles_df: Optional[pd.DataFrame],
    max_rows: int,
) -> list[dict[str, Any]]:
    if edges_df is None or edges_df.empty:
        return []

    df = edges_df.head(int(max_rows)).copy()
    items: list[dict[str, Any]] = []

    for r in df.itertuples(index=False):
        fk_table = r.fk_table
        fk_col = r.fk_column
        child_profile = _profile_lookup(profiles_df, fk_table, fk_col) if profiles_df is not None else {}

        item: dict[str, Any] = {
            "relationship": getattr(r, "relationship", f"{r.fk_table}.{r.fk_column} -> {r.pk_table}.{r.pk_column}"),
            "child": {"table": r.fk_table, "column": r.fk_column},
            "parent": {"table": r.pk_table, "column": r.pk_column},
            "type": {"cardinality": getattr(r, "cardinality", None), "optional": getattr(r, "optional", None)},
            "confidence": {
                "score": getattr(r, "score", None),
                "distinct_coverage": getattr(r, "distinct_coverage", None),
                "row_coverage": getattr(r, "row_coverage", None),
                "name_similarity": getattr(r, "name_sim", None),
                "range_penalty": getattr(r, "range_penalty", None),
                # if you surface soft rep evidence in edges_df later:
                "ind_mode": getattr(r, "mode", None),
                "fk_rep": getattr(r, "fk_rep", None),
                "pk_rep": getattr(r, "pk_rep", None),
            },
            "evidence": {
                "child_non_null_fk_rows": getattr(r, "fk_non_null_rows", None),
                "child_distinct_fk_values": getattr(r, "fk_distinct", None),
                "parent_distinct_pk_values": getattr(r, "pk_distinct", None),
                "intersection_distinct_values": getattr(r, "intersection_distinct", None),
                "matched_child_rows": getattr(r, "matched_rows", None),
                "child_total_rows": child_profile.get("n_rows", None),
                "child_null_fk_rows": child_profile.get("n_null", None),
                "child_null_fk_ratio": child_profile.get("null_ratio", None),
            },
        }

        # Remove null fields to keep payload smaller.
        item["confidence"] = {k: v for k, v in item["confidence"].items() if v is not None}
        item["evidence"] = {k: v for k, v in item["evidence"].items() if v is not None}

        items.append(_json_safe(item))

    return items


def _select_profile_base_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    keep = [c for c in BASE_PROFILE_COLS if c in df.columns]
    return df[keep].copy()


# =============================================================================
# Routes AKA Endpoints
# =============================================================================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/discover")
def discover(
    files: list[UploadFile] = File(...),
    include: Optional[str] = Query(
        None,
        description=(
            "Comma separated list of stages to return. "
            "Options -> relationships, profiles, profiles_enhanced, ucc, ucc_composite, ind, scored, rel_missing, "
            "duplicates, default_fk, surrogate_keys"
        ),
    ),
    include_stage: Optional[list[Stage]] = Query(
        None,
        description="Repeatable alternative to include. Example -> include_stage=relationships&include_stage=profiles",
    ),
    max_rows_per_stage: int = Query(
        500,
        ge=1,
        le=50_000,
        description="Maximum number of items returned per stage (payload only, not compute).",
    ),

    # -------------------------
    # IND mode switch
    # -------------------------
    ind_mode: IndMode = Query(
        "strict",
        description="IND discovery mode. strict -> exact matching. soft -> representation matching (padding, float->int, etc).",
    ),

    # -------------------------
    # Strict IND thresholds (strict mode only)
    # -------------------------
    min_distinct_coverage: float = Query(
        STRICT_DEFAULTS["min_distinct_coverage"],
        ge=0.0,
        le=1.0,
        description="Strict IND only -> minimum distinct coverage used during strict IND discovery.",
    ),
    min_row_coverage: float = Query(
        STRICT_DEFAULTS["min_row_coverage"],
        ge=0.0,
        le=1.0,
        description="Strict IND only -> minimum row coverage used during strict IND discovery (if enabled in strict mode).",
    ),

    min_non_null_rows: int = Query(
        20,
        ge=0,
        description="Minimum non null rows required for a column to be considered as an FK candidate (applies to both modes).",
    ),

    # -------------------------
    # Soft IND thresholds (soft mode only)
    # -------------------------
    soft_min_row_coverage: float = Query(
        SOFT_DEFAULTS["soft_min_row_coverage"],
        ge=0.0,
        le=1.0,
        description="Soft IND only -> minimum row coverage to accept a candidate.",
    ),
    soft_min_distinct_coverage: float = Query(
        SOFT_DEFAULTS["soft_min_distinct_coverage"],
        ge=0.0,
        le=1.0,
        description="Soft IND only -> minimum distinct coverage to accept a candidate.",
    ),
    soft_min_name_sim_for_routing: float = Query(
        SOFT_DEFAULTS["soft_min_name_sim_for_routing"],
        ge=0.0,
        le=1.0,
        description="Soft IND only -> routing threshold to limit FK->PK comparisons by name similarity.",
    ),
    soft_max_parents_per_fk: int = Query(
        SOFT_DEFAULTS["soft_max_parents_per_fk"],
        ge=1,
        le=500,
        description="Soft IND only -> cap number of parent candidates compared per FK column.",
    ),
    soft_small_domain_fk_distinct_max: int = Query(
        SOFT_DEFAULTS["soft_small_domain_fk_distinct_max"],
        ge=0,
        le=10_000,
        description="Soft IND only -> treat FK columns with <= this many distinct values as small-domain.",
    ),
    soft_small_domain_min_name_sim: float = Query(
        SOFT_DEFAULTS["soft_small_domain_min_name_sim"],
        ge=0.0,
        le=1.0,
        description="Soft IND only -> extra name similarity required for small-domain FK columns to reduce false positives.",
    ),

    # -------------------------
    # Optional checks (opt-in)
    # -------------------------
    rel_missing: bool = Query(False, description="If true -> run relational missingness checks for discovered relationships."),
    rel_missing_sample_n: int = Query(10, ge=1, le=200, description="How many example values and row indices to include per relationship."),
    rel_orphan_warn_ratio: float = Query(0.01, ge=0.0, le=1.0, description="Warning threshold for orphan FK ratio."),
    rel_orphan_fail_ratio: float = Query(0.05, ge=0.0, le=1.0, description="Fail threshold for orphan FK ratio."),
    rel_missing_warn_ratio: float = Query(0.05, ge=0.0, le=1.0, description="Warning threshold for missing children ratio."),
    rel_missing_fail_ratio: float = Query(0.20, ge=0.0, le=1.0, description="Fail threshold for missing children ratio."),

    duplicates: bool = Query(False, description="If true -> run duplicate checks."),

    default_fk: bool = Query(False, description="If true -> detect suspicious default placeholder values in discovered FK columns."),
    default_fk_sample_n: int = Query(5, ge=0, le=200, description="How many example default values to include per relationship (when available)."),

    # Composite UCC discovery is opt-in (can be expensive on wide tables).
    ucc_composite: bool = Query(False, description="If true -> discover composite unique column combinations (candidate composite keys)."),
    ucc_composite_max_k: int = Query(3, ge=2, le=5, description="Max number of columns in a composite key search (2..5)."),
    ucc_composite_max_cols_per_table: int = Query(12, ge=2, le=60, description="Max number of candidate columns per table to consider in composite search."),

    debug_bundle: bool = Query(False, description="If true and include is not set -> returns many stages (useful for debugging)."),

    include_viz: Optional[Literal["dot"]] = Query(None, description="Set to 'dot' -> include Graphviz DOT text for relationships."),

    # -------------------------
    # Selection threshold (final edge acceptance)
    # -------------------------
    min_relationship_score: float = Query(
        0.90,
        ge=0.0,
        le=1.0,
        description="Minimum final edge score required to accept a relationship.",
    ),
):
    # 1) Decide which stages to return
    requested = _parse_include(include_csv=include, include_list=include_stage, debug_bundle=debug_bundle)

    if "recommendations" in requested:
        # make dependencies inspectable in the response too
        for dep in ("rel_missing", "duplicates", "default_fk", "surrogate_keys", "ucc_composite"):
            if dep not in requested:
                requested.append(dep)

    stages_in_input: list[str] = []
    if include:
        stages_in_input = [p.strip().lower() for p in include.split(",") if p.strip()]
    elif include_stage:
        stages_in_input = [str(s) for s in include_stage]

    # 2) Validate parameter combinations early to prevent silent confusion
    if ind_mode == "strict":
        soft_non_default = {
            "soft_min_row_coverage": soft_min_row_coverage != SOFT_DEFAULTS["soft_min_row_coverage"],
            "soft_min_distinct_coverage": soft_min_distinct_coverage != SOFT_DEFAULTS["soft_min_distinct_coverage"],
            "soft_min_name_sim_for_routing": soft_min_name_sim_for_routing != SOFT_DEFAULTS["soft_min_name_sim_for_routing"],
            "soft_max_parents_per_fk": soft_max_parents_per_fk != SOFT_DEFAULTS["soft_max_parents_per_fk"],
            "soft_small_domain_fk_distinct_max": soft_small_domain_fk_distinct_max != SOFT_DEFAULTS["soft_small_domain_fk_distinct_max"],
            "soft_small_domain_min_name_sim": soft_small_domain_min_name_sim != SOFT_DEFAULTS["soft_small_domain_min_name_sim"],
        }
        provided = [k for k, changed in soft_non_default.items() if changed]
        if provided:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "INVALID_PARAM_COMBO",
                    "message": "Soft IND parameters were provided but ind_mode is strict.",
                    "details": {"ind_mode": "strict", "soft_params_changed": provided},
                },
            )

    if ind_mode == "soft":
        strict_non_default = {
            "min_distinct_coverage": min_distinct_coverage != STRICT_DEFAULTS["min_distinct_coverage"],
            "min_row_coverage": min_row_coverage != STRICT_DEFAULTS["min_row_coverage"],
        }
        provided = [k for k, changed in strict_non_default.items() if changed]
        if provided:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "INVALID_PARAM_COMBO",
                    "message": "Strict IND parameters were provided but ind_mode is soft.",
                    "details": {"ind_mode": "soft", "strict_params_changed": provided},
                },
            )

    # 3) Decide profiling mode once
    profile_mode: Literal["basic", "enhanced"] = (
        "enhanced" if (("profiles_enhanced" in requested) or ("recommendations" in requested)) else "basic"
    )

    # 4) Read files into DataFrames
    dfs = _read_uploads(files)

    # 5) Decide which optional computations are needed
    needs_rel_missing = bool(rel_missing or ("rel_missing" in requested))
    needs_duplicates = bool(duplicates or ("duplicates" in requested) or ("duplicates_summary" in requested))
    needs_default_fk = bool(default_fk or ("default_fk" in requested) or ("default_fk_summary" in requested))
    needs_surrogate = bool(("surrogate_keys" in requested) or ("surrogate_keys_summary" in requested))
    needs_ucc_composite = bool(ucc_composite or ("ucc_composite" in requested))

    # 6) Build pipeline config
    cfg = PipelineConfig(
        profile_mode=profile_mode,

        # selection
        min_edge_score=min_relationship_score,

        # IND mode
        ind_mode=ind_mode,
        ind_min_distinct_coverage=min_distinct_coverage,  # strict only
        ind_min_non_null_rows=min_non_null_rows,

        # If your strict mode uses row coverage, wire it here too (only if run.py supports it)
        # ind_min_row_coverage=min_row_coverage,

        # soft IND knobs (only used when ind_mode="soft")
        soft_ind_min_distinct_coverage=soft_min_distinct_coverage,
        soft_ind_min_row_coverage=soft_min_row_coverage,
        soft_ind_min_name_sim_for_routing=soft_min_name_sim_for_routing,
        soft_ind_max_parents_per_fk=soft_max_parents_per_fk,
        soft_ind_small_domain_fk_distinct_max=soft_small_domain_fk_distinct_max,
        soft_ind_small_domain_min_name_sim=soft_small_domain_min_name_sim,

        # optional checks
        rel_missing_enabled=needs_rel_missing,
        rel_missing_sample_n=rel_missing_sample_n,
        rel_orphan_warn_ratio=rel_orphan_warn_ratio,
        rel_orphan_fail_ratio=rel_orphan_fail_ratio,
        rel_missing_warn_ratio=rel_missing_warn_ratio,
        rel_missing_fail_ratio=rel_missing_fail_ratio,

        duplicates_enabled=needs_duplicates,

        default_fk_enabled=needs_default_fk,
        default_fk_sample_n=default_fk_sample_n,

        surrogate_keys_enabled=needs_surrogate,

        composite_ucc_enabled=needs_ucc_composite,
        composite_ucc_max_k=ucc_composite_max_k,
        composite_ucc_max_cols_per_table=ucc_composite_max_cols_per_table,
    )

    # 7) Run pipeline once
    t0 = time.time()
    out = run_schema_discovery(dfs, config=cfg, include_stages=requested)
    runtime_sec = round(time.time() - t0, 4)

    # 8) Build base response
    config_used: dict[str, Any] = {
        "profile_mode": profile_mode,
        "min_relationship_score": float(min_relationship_score),
        "min_non_null_rows": int(min_non_null_rows),
        "max_rows_per_stage": int(max_rows_per_stage),
        "debug_bundle": bool(debug_bundle),
        "rel_missing": bool(cfg.rel_missing_enabled),
        "rel_missing_sample_n": int(rel_missing_sample_n),
        "default_fk": bool(cfg.default_fk_enabled),
        "default_fk_sample_n": int(default_fk_sample_n),
        "duplicates": bool(cfg.duplicates_enabled),
        "surrogate_keys": bool(cfg.surrogate_keys_enabled),
        "ucc_composite": bool(cfg.composite_ucc_enabled),
        "ucc_composite_max_k": int(ucc_composite_max_k),
        "ucc_composite_max_cols_per_table": int(ucc_composite_max_cols_per_table),
        "ind_mode": str(ind_mode),
    }

    if ind_mode == "strict":
        config_used.update(
            {
                "strict_min_distinct_coverage": float(min_distinct_coverage),
                "strict_min_row_coverage": float(min_row_coverage),
            }
        )
    else:
        config_used.update(
            {
                "soft_min_distinct_coverage": float(soft_min_distinct_coverage),
                "soft_min_row_coverage": float(soft_min_row_coverage),
                "soft_min_name_sim_for_routing": float(soft_min_name_sim_for_routing),
                "soft_max_parents_per_fk": int(soft_max_parents_per_fk),
                "soft_small_domain_fk_distinct_max": int(soft_small_domain_fk_distinct_max),
                "soft_small_domain_min_name_sim": float(soft_small_domain_min_name_sim),
            }
        )

    response: dict[str, Any] = {
        "meta": {
            "runtime_sec": runtime_sec,
            "tables": {k: {"rows": int(len(v)), "cols": int(v.shape[1])} for k, v in dfs.items()},
            "requested_stages": requested,
            "config_used": config_used,
        },
        "results": {},
    }

    if stages_in_input:
        response["stages_in_input"] = {"detected": stages_in_input, "source": "include or include_stage param"}

    # 9) Safe stage wrapper
    def stage_block(stage_name: str, build_fn):
        try:
            return build_fn()
        except Exception as e:
            return {
                "count": 0,
                "items": [],
                "stage_error": {
                    "code": "STAGE_FAILED",
                    "message": f"Stage '{stage_name}' failed",
                    "details": {"exception_type": type(e).__name__, "exception_message": str(e)},
                },
            }

    results: dict[str, Any] = {}

    # 10) relationships
    if "relationships" in requested:

        def build_relationships():
            edges_df = out.get("edges_df", pd.DataFrame())
            profiles_df = out.get("profiles_df", pd.DataFrame())

            block: dict[str, Any] = {
                "summary": _summarise_relationships(edges_df),
                "count": int(len(edges_df)) if edges_df is not None else 0,
                "items": _edges_to_relationship_items(edges_df, profiles_df=profiles_df, max_rows=max_rows_per_stage),
            }

            if include_viz == "dot":
                block["viz"] = {"dot": edges_to_dot(edges_df, show_labels=True)}
            return block

        results["relationships"] = stage_block("relationships", build_relationships)

    # 11) profiles (base columns only)
    if "profiles" in requested:
        results["profiles"] = stage_block(
            "profiles",
            lambda: {
                "count": int(len(out.get("profiles_df", pd.DataFrame()))),
                "items": _df_to_records(
                    _select_profile_base_cols(out.get("profiles_df", pd.DataFrame())),
                    max_rows=max_rows_per_stage,
                ),
            },
        )

    # 12) ucc (unary)
    if "ucc" in requested:
        results["ucc"] = stage_block(
            "ucc",
            lambda: {
                "count": int(len(out.get("ucc_df", pd.DataFrame()))),
                "items": _df_to_records(out.get("ucc_df", pd.DataFrame()), max_rows=max_rows_per_stage),
            },
        )

    # 13) ucc_composite (opt-in)
    if "ucc_composite" in requested:
        results["ucc_composite"] = stage_block(
            "ucc_composite",
            lambda: {
                "count": int(len(out.get("composite_ucc_df", pd.DataFrame()))),
                "items": _df_to_records(out.get("composite_ucc_df", pd.DataFrame()), max_rows=max_rows_per_stage),
            },
        )

    # 14) ind
    if "ind" in requested:
        results["ind"] = stage_block(
            "ind",
            lambda: {
                "count": int(len(out.get("ind_df", pd.DataFrame()))),
                "items": _df_to_records(out.get("ind_df", pd.DataFrame()), max_rows=max_rows_per_stage),
            },
        )

    # 15) scored
    if "scored" in requested:
        results["scored"] = stage_block(
            "scored",
            lambda: {
                "count": int(len(out.get("scored_df", pd.DataFrame()))),
                "items": _df_to_records(out.get("scored_df", pd.DataFrame()), max_rows=max_rows_per_stage),
            },
        )

    # 16) rel_missing
    if "rel_missing" in requested:
        results["rel_missing"] = stage_block(
            "rel_missing",
            lambda: {
                "count": int(len(out.get("rel_missing_df", pd.DataFrame()))),
                "items": _df_to_records(out.get("rel_missing_df", pd.DataFrame()), max_rows=max_rows_per_stage),
            },
        )

    # 17) profiles_enhanced (full profiler output)
    if "profiles_enhanced" in requested:
        if profile_mode != "enhanced":
            results["profiles_enhanced"] = {
                "count": 0,
                "items": [],
                "stage_error": {
                    "code": "NOT_COMPUTED",
                    "message": "profiles_enhanced requested but profiler ran in basic mode",
                    "details": {"hint": "Request profiles_enhanced to enable enhanced mode"},
                },
            }
        else:
            results["profiles_enhanced"] = stage_block(
                "profiles_enhanced",
                lambda: {
                    "count": int(len(out.get("profiles_df", pd.DataFrame()))),
                    "items": _df_to_records(out.get("profiles_df", pd.DataFrame()), max_rows=max_rows_per_stage),
                },
            )

    # 18) duplicates
    if "duplicates" in requested or "duplicates_summary" in requested:

        def build_duplicates():
            dup_out = out.get("duplicates", {}) or {}
            summary = {k: int(len(v)) for k, v in dup_out.items() if isinstance(v, pd.DataFrame)}

            if "duplicates_summary" in requested and "duplicates" not in requested:
                return {"summary": summary, "count": int(sum(summary.values())), "items": []}

            items = {
                k: _df_to_records(v, max_rows=max_rows_per_stage)
                for k, v in dup_out.items()
                if isinstance(v, pd.DataFrame)
            }
            return {"summary": summary, "count": int(sum(summary.values())), "items": items}

        results["duplicates"] = stage_block("duplicates", build_duplicates)

    # 19) default_fk
    if "default_fk" in requested or "default_fk_summary" in requested:

        def build_default_fk():
            df = out.get("default_fk_df", pd.DataFrame())

            detected = (
                int((df.get("default_detected") == True).sum())  # noqa: E712
                if (df is not None and not df.empty and "default_detected" in df.columns)
                else 0
            )
            total = int(len(df)) if df is not None else 0
            summary = {"relationships_checked": total, "relationships_flagged": detected}

            if "default_fk_summary" in requested and "default_fk" not in requested:
                return {"summary": summary, "count": int(detected), "items": []}

            return {
                "summary": summary,
                "count": int(len(df)) if df is not None else 0,
                "items": _df_to_records(df, max_rows=max_rows_per_stage),
            }

        results["default_fk"] = stage_block("default_fk", build_default_fk)

    # 20) surrogate_keys
    if "surrogate_keys" in requested or "surrogate_keys_summary" in requested:

        def build_surrogate():
            out_sk = out.get("surrogate_keys", {}) or {}
            items = out_sk.get("items", []) or []

            flagged = sum(1 for it in items if it.get("is_surrogate") is True)
            summary = {"candidates": len(items), "flagged_surrogate": int(flagged)}

            if "surrogate_keys_summary" in requested and "surrogate_keys" not in requested:
                return {"summary": summary, "count": int(flagged), "items": []}

            return {"summary": summary, "count": len(items), "items": items[: int(max_rows_per_stage)]}

        results["surrogate_keys"] = stage_block("surrogate_keys", build_surrogate)

    # 21) recommendations
    if "recommendations" in requested:

        def build_recommendations():
            reco = out.get("recommendations", {}) or {}
            return {"count": 1, "items": [reco]}

        results["recommendations"] = stage_block("recommendations", build_recommendations)

    response["results"] = results
    return JSONResponse(content=_json_safe(response))
